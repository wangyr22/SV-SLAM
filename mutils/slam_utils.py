def export_kitti_pose(pose_list, out_path):
    """
    Export pose in kitti format. see https://github.com/MichaelGrupp/evo/wiki/Formats

    pose_list: iterable of 4x4 matrices.
    """
    out = []
    for i in range(len(pose_list)):
        out.append(
            ' '.join( [str(x) for x in pose_list[i].reshape(-1)[:12]] ) + '\n'
        )
    with open(out_path, 'w') as f:
        f.writelines(out)