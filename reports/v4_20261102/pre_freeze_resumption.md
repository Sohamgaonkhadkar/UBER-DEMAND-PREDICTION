# Pre-freeze development resumption

The local orchestrator started development when historical acquisition completed
while matched-data control code was being finalized. The initial run used the
earlier fixed model list. It was stopped **before selection freeze or any June
acquisition/evaluation**. Earlier per-month NPZ files, model checkpoints and
execution logs are preserved; copies of the interrupted logs are retained here.

The resumed run adds the declared expanding-history LightGBM and calendar
controls. Every preserved prediction path is reproduced exactly under the final
source before reuse (saved maximum differences must be zero). Merged results
use new `development_complete_*` files, leaving earlier partial results intact.
Only this complete set determines selection and its subsequent source freeze.
No final target informed this resumption or any model choice.
