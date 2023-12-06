def get_accuracy(model, dataset, top5=False):
    pass


def get_top1_accuracy(model, dataset):
    return get_accuracy(model, dataset, top5=False)


def get_top5_accuracy(model, dataset):
    return get_accuracy(model, dataset, top5=True)
