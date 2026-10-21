import os

# Directory holding the SAM 2.1 checkpoints, shared by every SAM 2 node. Override it with the SAM2_WEIGHTS_DIR
# environment variable, or point a node at one file with --sam-checkpoint.
SAM2_WEIGHTS_DIR = os.environ.get("SAM2_WEIGHTS_DIR", os.path.expanduser("~/model_weights/sam2"))

# --sam-model choice -> (checkpoint file in SAM2_WEIGHTS_DIR, model config inside the sam2 package)
SAM2_MODELS = {
    "large": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
    "small": ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
}


def add_sam2_args(parser):
    """Add the --sam-model, --sam-checkpoint and --sam-config options to an argparse parser."""
    group = parser.add_argument_group("SAM 2 model")
    group.add_argument("--sam-model", choices=SAM2_MODELS, default="small",
                       help=f"SAM 2.1 model size, loaded from {SAM2_WEIGHTS_DIR} (default: small).")
    group.add_argument("--sam-checkpoint", default=None,
                       help="Checkpoint file, overriding the one chosen by --sam-model.")
    group.add_argument("--sam-config", default=None,
                       help="Model config inside the sam2 package, overriding the one chosen by --sam-model.")


def resolve_sam2_model(args):
    """Return (checkpoint_path, model_config) for args parsed with add_sam2_args."""
    checkpoint_file, model_config = SAM2_MODELS[args.sam_model]
    checkpoint_path = args.sam_checkpoint or os.path.join(SAM2_WEIGHTS_DIR, checkpoint_file)
    return checkpoint_path, args.sam_config or model_config
