import argparse
from huggingface_hub import snapshot_download, login

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_token', type=str, help='Your HuggingFace access token')
    args = parser.parse_args()

    login(token=args.hf_token)
    snapshot_download(
        repo_type='dataset',
        repo_id="TencentARC/Video-Holmes",
        local_dir="./Benchmark"
    )

if __name__ == "__main__":
    main()