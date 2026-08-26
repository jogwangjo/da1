"""HTTP Range 요청으로 원격 zip에서 일부 엔트리만 추출."""
import struct
import urllib.request
import zlib

CD_FMT = "<IHHHHHHIIIHHHHHII"  # central directory 고정부 46B


class RemoteZip:
    def __init__(self, url):
        self.url = url
        size = self._head_size()
        tail = self._range(max(0, size - 66000), size - 1)
        eocd = tail.rfind(b"PK\x05\x06")
        if eocd < 0:
            raise RuntimeError("EOCD not found")
        cnt, cd_size, cd_off = struct.unpack("<HII", tail[eocd + 10 : eocd + 20])
        if cd_off + cd_size > size or cnt == 0xFFFF:  # zip64 경로
            loc = tail.rfind(b"PK\x06\x07")
            if loc < 0:
                raise RuntimeError("zip64 locator not found")
            z64_off = struct.unpack("<Q", tail[loc + 8 : loc + 16])[0]
            rec = self._range(z64_off, z64_off + 55)
            cnt = struct.unpack("<Q", rec[32:40])[0]
            cd_size = struct.unpack("<Q", rec[40:48])[0]
            cd_off = struct.unpack("<Q", rec[48:56])[0]
        cd = self._range(cd_off, cd_off + cd_size - 1)
        self.entries = {}
        pos = 0
        for _ in range(cnt):
            f = struct.unpack(CD_FMT, cd[pos : pos + 46])
            (sig, _vm, _vn, _flags, method, _mt, _md, _crc,
             csize, usize, nlen, elen, clen, _disk, _ia, _ea, lho) = f
            name = cd[pos + 46 : pos + 46 + nlen].decode("utf-8", "replace")
            extra = cd[pos + 46 + nlen : pos + 46 + nlen + elen]
            if csize == 0xFFFFFFFF or usize == 0xFFFFFFFF or lho == 0xFFFFFFFF:
                vals = self._zip64_extra(extra)
                i = 0
                if usize == 0xFFFFFFFF:
                    usize = vals[i]; i += 1
                if csize == 0xFFFFFFFF:
                    csize = vals[i]; i += 1
                if lho == 0xFFFFFFFF:
                    lho = vals[i]; i += 1
            self.entries[name] = dict(method=method, csize=csize,
                                      usize=usize, lho=lho)
            pos += 46 + nlen + elen + clen

    @staticmethod
    def _zip64_extra(extra):
        e = 0
        while e + 4 <= len(extra):
            hid, hsz = struct.unpack("<HH", extra[e : e + 4])
            if hid == 1:
                return list(struct.unpack(
                    "<" + "Q" * (hsz // 8), extra[e + 4 : e + 4 + hsz]))
            e += 4 + hsz
        return []

    def _head_size(self):
        req = urllib.request.Request(self.url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            return int(r.headers["Content-Length"])

    def _range(self, start, end):
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req) as r:
            return r.read()

    def namelist(self):
        return list(self.entries.keys())

    def read(self, name):
        ent = self.entries[name]
        hint = ent["lho"]
        head = self._range(hint, hint + 512)
        if not head.startswith(b"PK\x03\x04"):
            raise RuntimeError(f"local header mismatch at {hint} for {name}")
        nlen, elen = struct.unpack("<HH", head[26:30])
        data_start = hint + 30 + nlen + elen
        blob = self._range(data_start, data_start + ent["csize"] - 1)
        if ent["method"] == 0:
            return blob
        if ent["method"] == 8:
            return zlib.decompress(blob, -15)
        raise RuntimeError(f"unsupported method {ent['method']}")

    def extract_matching(self, predicate, limit, dest):
        from pathlib import Path

        out = []
        for n in self.namelist():
            if len(out) >= limit:
                break
            if predicate(n):
                tgt = Path(dest) / Path(n).name
                if not tgt.exists():
                    tgt.write_bytes(self.read(n))
                    print("got", tgt.name, round(tgt.stat().st_size / 1e6, 1), "MB")
                out.append(tgt)
        return out
