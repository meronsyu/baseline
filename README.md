# baseline
8/10までで完成しているベースラインです。

<img width="1210" height="600" alt="image" src="https://github.com/user-attachments/assets/e5bf2573-c143-47bb-9735-d84888a5c986" />

<コードの構成>
baseline
├── [ 4.0K]  eval_aime
│   ├── [  344]  aime_prediction.sh
│   ├── [ 1.8K]  aime_script.sh
│   ├── [ 4.0K]  conf
│   ├── [ 4.0K]  hle_benchmark
│   │   
│   ├── [ 4.0K]  judged
│   ├── [ 4.0K]  leaderboard
│   ├── [ 4.0K]  notebooks
│   ├── [ 4.0K]  outputs
│   ├── [ 4.0K]  predictions
│   └── [  118]  requirements.txt
├── [ 4.0K]  eval_hle
│   ├── [ 4.0K]  conf
│   ├── [ 4.0K]  hle_benchmark
│   │   
│   ├── [  344]  hle_prediction.sh
│   ├── [ 1.8K]  hle_script.sh
│   ├── [ 4.0K]  judged
│   ├── [ 4.0K]  leaderboard
│   ├── [ 4.0K]  notebooks
│   ├── [ 4.0K]  outputs
│   ├── [ 4.0K]  predictions
│   └── [  118]  requirements.txt
└── [ 4.0K]  train
    ├── [ 3.9K]  qwen3-32b_grpo.sh
    └── [ 5.0K]  qwen3-32b_ssh.sh

AIMEを使ってQwen3 32Bを作るコード
HLEを使ってomni3で評価するコード
./hle_script.shについて、huggingfaceのトークンと、OpenAPIについて書いてください。

権限のために.shスクリプトに権限を与えてください。Home directory

chmod +x ~/baseline/train/qwen3-32b_grpo.sh  
chmod +x ~/baseline/train/qwen3-32b_ssh.sh
chmod +x ~/baseline/eval_hle/hle_prediction.sh
chmod +x ~/baseline/eval_hle/hle_script.sh
chmod +x ~/baseline/eval_aime/aime_prediction.sh
chmod +x ~/baseline/eval_aime/aime_script.sh

sftのためのコード

sequenceで、
sft -> AIME 25
sft -> HLE
sft -> grpo -> HLE
grpo -> HLE
を作りたいですが、どなたか作ってください