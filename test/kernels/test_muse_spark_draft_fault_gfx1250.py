"""Compile every gfx1250 Draft kernel with mailbox fault injection enabled."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip
        and "gfx1250" in torch.cuda.get_device_properties(0).gcnArchName
    ),
    reason="needs a gfx1250 GPU",
)


@pytest.mark.parametrize("rows", [8, 16, 32, 64])
def test_gfx1250_draft_fault_variants_compile(rows):
    from sgl_amd_mono.muse_spark.gfx1250 import draft_attn as DA
    from sgl_amd_mono.muse_spark.gfx1250 import draft_attn_half as DH
    from sgl_amd_mono.muse_spark.gfx1250 import draft_mlp as DM
    from sgl_amd_mono.muse_spark.gfx1250.runner import compile_only

    launches = (
        DH.build_dpre(
            DH.DPreBuild(tokens=rows, fault_site=1, fault_after=1)
        ),
        DA.build_dattn(
            DA.DAttnBuild(
                tokens=rows,
                fault_site=1,
                fault_after=1,
                resident_blocks=96,
            )
        ),
        DH.build_dpost(
            DH.DPostBuild(
                tokens=rows, npes=4, fault_site=1, fault_after=1
            )
        ),
        DM.build_dmlp(
            DM.DMLPBuild(
                tokens=rows, npes=4, fault_site=1, fault_after=1
            )
        ),
    )
    for launch in launches:
        compile_only(launch)
