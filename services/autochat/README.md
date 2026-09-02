# Autochat Service for MinoriBot

A standalone process that connects to the robot-side RPC via an aiorpcx WebSocket.

### Usage

Modify your configuration `config/chat/autochat.yaml`

In the root directory of the MinoriBot repository:

```bash
pip install -r services/autochat/requirements.txt
python -m services.autochat.serve
```
