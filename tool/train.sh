python3 -m torch.distributed.launch --nproc_per_node=1 main.py --src_dataset front3d --trgt_dataset scannet --sam_path "/cluster/personal/DODA_sam/pretrained_model/sam_vit_b_01ec64.pth"

