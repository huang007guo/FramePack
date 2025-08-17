bucket_options = {
    640: [
        (416, 960),
        (448, 864),
        (480, 832),
        (512, 768),
        (544, 704),
        (576, 672),
        (608, 640),
        (640, 608),
        (672, 576),
        (704, 544),
        (768, 512),
        (832, 480),
        (864, 448),
        (960, 416),
    ],
    768: [
        (480, 1152),
        (512, 1024),
        (576, 896),
        (640, 832),
        (704, 768),
        (768, 704),
        (832, 640),
        (896, 576),
        (1024, 512),
        (1152, 480),
    ],
    1024: [
        (512, 1536),
        (576, 1408),
        (640, 1280),
        (704, 1152),
        (768, 1024),
        (832, 896),
        (896, 832),
        (1024, 768),
        (1152, 704),
        (1280, 640),
        (1408, 576),
        (1536, 512),
    ]
}


def find_nearest_bucket(h, w, resolution=640):
    min_metric = float('inf')
    best_bucket = None
    for (bucket_h, bucket_w) in bucket_options[resolution]:
        metric = abs(h * bucket_w - w * bucket_h)
        if metric <= min_metric:
            min_metric = metric
            best_bucket = (bucket_h, bucket_w)
    return best_bucket