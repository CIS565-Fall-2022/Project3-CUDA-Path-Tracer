cmake_minimum_required(VERSION 3.1)
set(headers
    "common.h"
    "efficient.h"
    )

set(sources
    "common.cu"
    "efficient.cu"
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

cuda_add_library(stream_compaction ${sources} ${headers})
