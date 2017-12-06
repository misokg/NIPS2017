import miso_rosenbrock

__author__ = 'jialeiwang'
__author__ = 'matthiaspoloczek'

def identify_problem(argv, bucket):
    benchmark_name = argv[0]
    decode = benchmark_name.split("_")
    if decode[0] == "miso":
        if decode[1] == "rb":
            problem_class = miso_rosenbrock.class_collection[benchmark_name]
            which_rb = int(argv[1])
            if decode[2] == "hyper":
                problem = problem_class(which_rb, bucket)
            else:
                replication_no = int(argv[2])
                problem = problem_class(replication_no, which_rb, bucket)
        else:
            raise ValueError("func name not recognized")
    else:
        raise ValueError("task name not recognized")
    return problem
