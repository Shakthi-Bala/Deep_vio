from train_common import parse_common_args, train_model

if __name__ == "__main__":
    args = parse_common_args(default_mode="vio", default_out="checkpoints/vio")
    train_model(args)
