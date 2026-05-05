from train_common import parse_common_args, train_model

DATA_DIR = "/home/sbalamurugan/cv_p4/blender_splats/new_data_out"

if __name__ == "__main__":
    args = parse_common_args(default_mode="imu", default_out="checkpoints/imu")
    if args.data_dir == "data/synthetic":
        args.data_dir = DATA_DIR
    train_model(args)
