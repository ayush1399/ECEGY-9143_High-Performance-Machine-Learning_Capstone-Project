import torch


def get_performance(model, dataloader, input_shape=(3, 224, 224)):
    """
    Returns the inference time of a PyTorch model in milliseconds.
    """
    if not torch.cuda.is_available():
        raise Exception("CUDA not available.")

    device = torch.device("cuda")
    model = model.to(device)
    model.eval()

    # Warm-up
    inputs = torch.randn(32, *input_shape).to(device)
    for _ in range(32):
        _ = model(inputs)

    # Initialize CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    total_time = 0.0
    num_samples = 0

    for inputs, _ in dataloader:
        inputs = inputs.to(device)
        num_samples += inputs.size(0)

        torch.cuda.synchronize()
        start_event.record()
        with torch.no_grad():
            _ = model(inputs)
        end_event.record()
        torch.cuda.synchronize()

        # Calculate the time taken and accumulate
        batch_time = start_event.elapsed_time(end_event)
        total_time += batch_time

        total_time_seconds = total_time / 1000
        images_per_second = num_samples / total_time_seconds

    return total_time / num_samples, images_per_second
