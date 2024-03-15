femnist_target_mac = 15e6
femnist_th_incre = 0.01
cifar_th_incre = 0.2
femnist_model_layers = {
    "stem.0": {
        "ansestors": [],
        "siblings": [],
        "descendants": ["stem.1"]
    },
    "stem.1": {
        "ansestors": ["stem.0"],
        "siblings": [],
        "descendants": ["cells.0.conv_a.op.1", "cells.0.downsample.1.0"]
    },
    "cells.0.conv_a.op.1": {
        "ansestors": ["stem.1"],
        "siblings": [],
        "descendants": ["cells.0.conv_a.op.2"]
    },
    "cells.0.conv_a.op.2": {
        "ansestors": ["cells.0.conv_a.op.1"],
        "siblings": [],
        "descendants": ["cells.0.conv_b.op.1"]
    },
    "cells.0.conv_b.op.1": {
        "ansestors": ["cells.0.conv_a.op.2"],
        "siblings": [],
        "descendants": ["cells.0.conv_b.op.2"]
    },
    "cells.0.conv_b.op.2": {
        "ansestors": ["cells.0.conv_b.op.1"],
        "siblings": ["cells.0.downsample.1.2.2.2.2.2"],
        "descendants": ["cells.1.conv_a.op.1", "cells.1.downsample.1.0"]
    },
    "cells.0.downsample.1.0": {
        "ansestors": ["stem.1"],
        "siblings": [],
        "descendants": ["cells.0.downsample.1.1"]
    },
    "cells.0.downsample.1.1": {
        "ansestors": ["cells.0.downsample.1.0"],
        "siblings": [],
        "descendants": ["cells.0.downsample.1.2.0"]
    },
    "cells.0.downsample.1.2.0": {
        "ansestors": ["cells.0.downsample.1.1"],
        "siblings": [],
        "descendants": ["cells.0.downsample.1.2.1"]
    },
    "cells.0.downsample.1.2.1": {
        "ansestors": ["cells.0.downsample.1.2.0"],
        "siblings": [],
        "descendants": ["cells.0.downsample.1.2.2.0"]
    },
    "cells.0.downsample.1.2.2.0": {
        "ansestors": ["cells.0.downsample.1.2.1"],
        "siblings": [],
        "descendants": ["cells.0.downsample.1.2.2.1"]
    },
    "cells.0.downsample.1.2.2.1": {
        "ansestors": ["cells.0.downsample.1.2.2.0"],
        "siblings": [],
        "descendants": ["cells.0.downsample.1.2.2.2.0"]
    },
    "cells.0.downsample.1.2.2.2.0": {
        "ansestors": ["cells.0.downsample.1.2.2.1"],
        "siblings": [],
        "descendants": ["cells.0.downsample.1.2.2.2.1"]
    },
    "cells.0.downsample.1.2.2.2.1": {
        "ansestors": ["cells.0.downsample.1.2.2.2.0"],
        "siblings": [],
        "descendants": ["cells.0.downsample.1.2.2.2.2.0"]
    },
    "cells.0.downsample.1.2.2.2.2.0": {
        "ansestors": ["cells.0.downsample.1.2.2.2.1"],
        "siblings": [],
        "descendants": ["cells.0.downsample.1.2.2.2.2.1"]
    },
    "cells.0.downsample.1.2.2.2.2.1": {
        "ansestors": ["cells.0.downsample.1.2.2.2.2.0"],
        "siblings": [],
        "descendants": ["cells.0.downsample.1.2.2.2.2.2"]
    },
    "cells.0.downsample.1.2.2.2.2.2": {
        "ansestors": ["cells.0.downsample.1.2.2.2.2.1"],
        "siblings": ["cells.0.conv_b.op.2"],
        "descendants": ["cells.1.conv_a.op.1", "cells.1.downsample.1.0"]
    },
    "cells.1.conv_a.op.1": {
        "ansestors": ["cells.0.conv_b.op.2", "cells.0.downsample.1.2.2.2.2.2"],
        "siblings": [],
        "descendants": ["cells.1.conv_a.op.2"]
    },
    "cells.1.conv_a.op.2": {
        "ansestors": ["cells.1.conv_a.op.1"],
        "siblings": [],
        "descendants": ["cells.1.conv_b.op.1"]
    },
    "cells.1.conv_b.op.1": {
        "ansestors": ["cells.1.conv_a.op.2"],
        "siblings": [],
        "descendants": ["cells.1.conv_b.op.2"]
    },
    "cells.1.conv_b.op.2": {
        "ansestors": ["cells.1.conv_b.op.1"],
        "siblings": ["cells.1.downsample.1.2.2.2.2.2"],
        "descendants": ["lastact.0"]
    },
    "cells.1.downsample.1.0": {
        "ansestors": ["cells.0.conv_b.op.2", "cells.0.downsample.1.2.2.2.2.2"],
        "siblings": [],
        "descendants": ["cells.1.downsample.1.1"]
    },
    "cells.1.downsample.1.1": {
        "ansestors": ["cells.1.downsample.1.0"],
        "siblings": [],
        "descendants": ["cells.1.downsample.1.2.0"]
    },
    "cells.1.downsample.1.2.0": {
        "ansestors": ["cells.1.downsample.1.1"],
        "siblings": [],
        "descendants": ["cells.1.downsample.1.2.1"]
    },
    "cells.1.downsample.1.2.1": {
        "ansestors": ["cells.1.downsample.1.2.0"],
        "siblings": [],
        "descendants": ["cells.1.downsample.1.2.2.0.0"]
    },
    "cells.1.downsample.1.2.2.0.0": {
        "ansestors": ["cells.1.downsample.1.2.1"],
        "siblings": [],
        "descendants": ["cells.1.downsample.1.2.2.0.1"]
    },
    "cells.1.downsample.1.2.2.0.1": {
        "ansestors": ["cells.1.downsample.1.2.2.0.0"],
        "siblings": [],
        "descendants": ["cells.1.downsample.1.2.2.0.2"]
    },
    "cells.1.downsample.1.2.2.0.2": {
        "ansestors": ["cells.1.downsample.1.2.2.0.1"],
        "siblings": [],
        "descendants": ["cells.1.downsample.1.2.2.1"]
    },
    "cells.1.downsample.1.2.2.1": {
        "ansestors": ["cells.1.downsample.1.2.2.0.2"],
        "siblings": [],
        "descendants": ["cells.1.downsample.1.2.2.2.0"]
    },
    "cells.1.downsample.1.2.2.2.0": {
        "ansestors": ["cells.1.downsample.1.2.2.1"],
        "siblings": [],
        "descendants": ["cells.1.downsample.1.2.2.2.1"]
    },
    "cells.1.downsample.1.2.2.2.1": {
        "ansestors": ["cells.1.downsample.1.2.2.2.0"],
        "siblings": [],
        "descendants": ["cells.1.downsample.1.2.2.2.2.0"]
    },
    "cells.1.downsample.1.2.2.2.2.0": {
        "ansestors": ["cells.1.downsample.1.2.2.2.1"],
        "siblings": [],
        "descendants": ["cells.1.downsample.1.2.2.2.2.1"]
    },
    "cells.1.downsample.1.2.2.2.2.1": {
        "ansestors": ["cells.1.downsample.1.2.2.2.2.0"],
        "siblings": [],
        "descendants": ["cells.1.downsample.1.2.2.2.2.2"]
    },
    "cells.1.downsample.1.2.2.2.2.2": {
        "ansestors": ["cells.1.downsample.1.2.2.2.2.1"],
        "siblings": ["cells.1.conv_b.op.2"],
        "descendants": ["lastact.0"]
    },
    "lastact.0": {
        "ansestors": ["cells.1.conv_b.op.2", "cells.1.downsample.1.2.2.2.2.2"],
        "siblings": [],
        "descendants": ["classifier"]
    },
    "classifier": {
        "ansestors": ["lastact.0"],
        "siblings": [],
        "descendants": []
    }
}