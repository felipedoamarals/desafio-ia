from typing import Dict, Any, Optional
from gridfs import GridFS
from pymongo import MongoClient, ASCENDING
from bson import ObjectId
import datetime as dt

class MongoRepo:
    """Repositório MongoDB para armazenar FITS via GridFS + metadados em coleção."""
    def __init__(self, uri: str = "mongodb://mongodb:27017", db_name: str = "desafio_ia"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.fs = GridFS(self.db, collection="frames_fs")
        self.frames = self.db["frames_meta"]
        self.frames.create_index([("frame_idx", ASCENDING)])
        self.frames.create_index([("ts_ms", ASCENDING)])

    def save_frame(self, fits_bytes: bytes, metadata: Dict[str, Any]) -> str:
        """Salva o arquivo FITS no GridFS e registra metadados em `frames_meta`."""
        file_id = self.fs.put(fits_bytes, filename=metadata.get("filename", "frame.fits"))
        doc = {**metadata, "file_id": file_id, "created_at": dt.datetime.utcnow()}
        res = self.frames.insert_one(doc)
        return str(res.inserted_id)

    def load_frame(self, meta_id: str) -> Optional[bytes]:
        doc = self.frames.find_one({"_id": ObjectId(meta_id)})
        if not doc:
            return None
        fh = self.fs.get(doc["file_id"]) 
        return fh.read()