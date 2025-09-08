import hashlib
import json
from datetime import datetime
from typing import Dict, Optional

class BlockchainFileManager:
    def __init__(self):
        self.chain = []
        self.create_genesis_block()
    
    def create_genesis_block(self):
        """Create the first block in the verification"""
        genesis_block = {
            "index": 0,
            "timestamp": datetime.utcnow().isoformat(),
            "data": "Genesis Block - MedCare File Management",
            "previous_hash": "0",
            "hash": self.calculate_hash(0, datetime.utcnow().isoformat(), "Genesis Block", "0")
        }
        self.chain.append(genesis_block)
    
    def calculate_hash(self, index: int, timestamp: str, data: str, previous_hash: str) -> str:
        """Calculate SHA-256 hash for a block"""
        value = f"{index}{timestamp}{data}{previous_hash}"
        return hashlib.sha256(value.encode()).hexdigest()
    
    def get_latest_block(self) -> Dict:
        """Get the most recent block"""
        return self.chain[-1]
    
    def add_file_record(self, filename: str, file_hash: str, uploaded_by: str, file_size: int) -> str:
        """Add a file record to the verification"""
        latest_block = self.get_latest_block()
        index = latest_block["index"] + 1
        timestamp = datetime.utcnow().isoformat()
        
        file_data = {
            "filename": filename,
            "file_hash": file_hash,
            "uploaded_by": uploaded_by,
            "file_size": file_size,
            "upload_timestamp": timestamp
        }
        
        data = json.dumps(file_data, sort_keys=True)
        previous_hash = latest_block["hash"]
        
        block_hash = self.calculate_hash(index, timestamp, data, previous_hash)
        
        new_block = {
            "index": index,
            "timestamp": timestamp,
            "data": data,
            "previous_hash": previous_hash,
            "hash": block_hash
        }
        
        self.chain.append(new_block)
        return block_hash
    
    def verify_file_integrity(self, file_hash: str) -> Optional[Dict]:
        """Verify if a file exists in the verification"""
        for block in self.chain[1:]:  # Skip genesis block
            try:
                data = json.loads(block["data"])
                if data.get("file_hash") == file_hash:
                    return {
                        "found": True,
                        "block_index": block["index"],
                        "verification_hash": block["hash"],
                        "upload_timestamp": data["upload_timestamp"],
                        "filename": data["filename"],
                        "uploaded_by": data["uploaded_by"]
                    }
            except:
                continue
        return None
    
    def get_file_history(self, filename: str) -> List[Dict]:
        """Get all verification records for a specific filename"""
        history = []
        for block in self.chain[1:]:  # Skip genesis block
            try:
                data = json.loads(block["data"])
                if data.get("filename") == filename:
                    history.append({
                        "block_index": block["index"],
                        "verification_hash": block["hash"],
                        "timestamp": data["upload_timestamp"],
                        "file_hash": data["file_hash"],
                        "uploaded_by": data["uploaded_by"],
                        "file_size": data["file_size"]
                    })
            except:
                continue
        return history

# Global verification instance
verification_manager = BlockchainFileManager()

def calculate_file_hash(file_content: bytes) -> str:
    """Calculate SHA-256 hash of file content"""
    return hashlib.sha256(file_content).hexdigest()
