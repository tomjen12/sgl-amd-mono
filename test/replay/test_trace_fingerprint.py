import copy

from trace_fingerprint import compare_fingerprints, fingerprint_trace


def _event(name, category, timestamp, duration=1, **args):
    return {
        "name": name,
        "cat": category,
        "ph": "X",
        "ts": timestamp,
        "dur": duration,
        "pid": 10,
        "tid": 20,
        "args": args,
    }


def _trace():
    return {
        "traceEvents": [
            _event("muse_replay/target/layer_0", "user_annotation", 100, 100),
            _event(
                "aten::linear",
                "cpu_op",
                110,
                10,
                **{"External id": 7, "Input Dims": [[8, 8192], [8192, 4096]]},
            ),
            _event(
                "gemm_0xabc_kernel.0123456789abcdef",
                "kernel",
                130,
                20,
                **{
                    "External id": 7,
                    "grid": [64, 1, 1],
                    "block": [256, 1, 1],
                },
            ),
        ]
    }


def test_fingerprint_normalizes_run_specific_fields():
    first = fingerprint_trace(_trace())
    changed = copy.deepcopy(_trace())
    changed["traceEvents"][2]["name"] = "gemm_0xdef_kernel.fedcba9876543210"
    changed["traceEvents"][2]["pid"] = 99
    changed["traceEvents"][2]["tid"] = 88
    changed["traceEvents"][2]["args"]["correlation"] = 123456
    second = fingerprint_trace(changed)

    assert compare_fingerprints(first, second) == []
    assert first["scopes"][0]["gpu_kernels"][0]["cpu_parent"] == "aten::linear"


def test_fingerprint_reports_launch_geometry_difference():
    expected = fingerprint_trace(_trace())
    changed = copy.deepcopy(_trace())
    changed["traceEvents"][2]["args"]["block"] = [128, 1, 1]
    actual = fingerprint_trace(changed)

    failures = compare_fingerprints(expected, actual)

    assert len(failures) == 1
    assert "gpu_kernels[0]" in failures[0]


def test_fingerprint_reports_operator_order_difference():
    expected_trace = _trace()
    expected_trace["traceEvents"].insert(
        2, _event("aten::silu", "cpu_op", 121, 2, **{"External id": 8})
    )
    actual_trace = copy.deepcopy(expected_trace)
    actual_trace["traceEvents"][1]["ts"] = 122
    expected = fingerprint_trace(expected_trace)
    actual = fingerprint_trace(actual_trace)

    failures = compare_fingerprints(expected, actual)

    assert len(failures) == 1
    assert "cpu_ops[0]" in failures[0]


def test_graph_latency_tolerance_is_opt_in():
    expected = fingerprint_trace(_trace())
    slower = copy.deepcopy(_trace())
    slower["traceEvents"][2]["dur"] = 30
    actual = fingerprint_trace(slower)

    assert compare_fingerprints(expected, actual) == []
    failures = compare_fingerprints(
        expected, actual, gpu_time_tolerance=0.10
    )
    assert len(failures) == 1
    assert "gpu_span_us" in failures[0]


def test_collective_payload_and_hip_clone_are_normalized():
    trace = _trace()
    trace["traceEvents"].insert(
        2,
        _event(
            "record_param_comms",
            "cpu_op",
            120,
            2,
            **{
                "collective_name": "all_reduce",
                "dtype": "BFloat16",
                "in_msg_nelems": 65536,
                "group_size": 4,
                "External id": 8,
            },
        ),
    )
    trace["traceEvents"][3]["name"] += " [clone .kd]"

    fingerprint = fingerprint_trace(trace)

    collective = fingerprint["scopes"][0]["cpu_ops"][1]
    assert collective["args"]["collective_name"] == "all_reduce"
    assert collective["args"]["input_elements"] == 65536
    assert "[clone" not in fingerprint["scopes"][0]["gpu_kernels"][0]["name"]
