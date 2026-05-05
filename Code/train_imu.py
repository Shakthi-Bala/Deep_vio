from train_common import parse_common_args, train_model

if __name__ == "__main__":
    args = parse_common_args(default_mode="imu", default_out="checkpoints/imu")
    train_model(args)
