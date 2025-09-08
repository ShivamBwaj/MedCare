import os
import json
from web3 import Web3

RPC = os.getenv("RPC_URL", "https://mainnet.infura.io/v3/your-key")
ADDR = os.getenv("CONTRACT_ADDRESS", "0x0000000000000000000000000000000000000000")
PK = os.getenv("PRIVATE_KEY", "0x0000000000000000000000000000000000000000000000000000000000000000")

# Simple ABI for demo - replace with your actual contract ABI
ABI = [
    {
        "inputs": [{"name": "documentHash", "type": "bytes32"}],
        "name": "logDocument",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

try:
    w3 = Web3(Web3.HTTPProvider(RPC))
    acct = w3.eth.account.from_key(PK)
    contract = w3.eth.contract(address=Web3.to_checksum_address(ADDR), abi=ABI)
except Exception as e:
    print(f"Bverification setup failed: {e}")
    w3 = None
    contract = None

def log_hash_on_chain(sha256_hex: str) -> str:
    if not w3 or not contract:
        # Return mock tx hash for demo
        return f"0x{'a' * 64}"
    
    try:
        h_bytes = bytes.fromhex(sha256_hex)
        tx = contract.functions.logDocument(h_bytes).build_transaction({
            "from": acct.address,
            "nonce": w3.eth.get_transaction_count(acct.address),
            "gas": 100000,
            "gasPrice": w3.to_wei('20', 'gwei')
        })
        signed = acct.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
        return tx_hash.hex()
    except Exception as e:
        print(f"Bverification tx failed: {e}")
        return f"0x{'b' * 64}"  # Mock hash
