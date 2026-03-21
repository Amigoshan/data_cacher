import torch

def gen_event_tensor_torch( events, num_bins = 15, width = 640, height = 640, device = "cuda"):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    :param device: device to use to perform computations
    :return voxel_grid: PyTorch event tensor (on the device specified)
    """
    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    if events.shape[0] == 0:
        return torch.zeros(num_bins, height, width, dtype=torch.float32, device=device)

    with torch.no_grad():

        events_torch = torch.from_numpy(events)
        events_torch = events_torch.to(device)

        voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32, device=device).flatten()

        # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = events_torch[-1, 0]
        first_stamp = events_torch[0, 0]
        deltaT = last_stamp - first_stamp

        if deltaT == 0:
            deltaT = 1.0

        # events_torch[:, 0] = (num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
        ts = (num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT #events_torch[:, 0]
        xs = events_torch[:, 1].long()
        ys = events_torch[:, 2].long()
        pols = events_torch[:, 3].float()
        pols[pols == 0] = -1  # polarity should be +1 / -1

        tis = torch.floor(ts)
        tis_long = tis.long()
        dts = ts - tis
        vals_left = pols * (1.0 - dts.float())
        vals_right = pols * dts.float()

        valid_indices = tis < num_bins
        valid_indices &= tis >= 0
        voxel_grid.index_add_(dim=0,
                            index=xs[valid_indices] + ys[valid_indices]
                            * width + tis_long[valid_indices] * width * height,
                            source=vals_left[valid_indices])

        valid_indices = (tis + 1) < num_bins
        valid_indices &= tis >= 0

        voxel_grid.index_add_(dim=0,
                            index=xs[valid_indices] + ys[valid_indices] * width
                            + (tis_long[valid_indices] + 1) * width * height,
                            source=vals_right[valid_indices])

        voxel_grid = voxel_grid.view(num_bins, height, width)

    return voxel_grid
