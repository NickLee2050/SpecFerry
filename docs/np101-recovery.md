# NP101 device recovery

Use this guide before board execution. EP (PCIe endpoint) runs on the host; SoC
(system on chip) runs on the board. See [getting started](getting-started.md) for
supported commands and [validation status](np101-status.md) for accepted configurations.
Past successes and automatic driver recovery do not establish current device health.

## Before and after execution

- Read the complete boot kernel log, board boot identity, recovery marker and device
  ownership. Confirm that the previous process exited and driver memory allocations
  returned to their recorded baseline. A clean log tail alone is insufficient.
- Use fixed binary, compiled graph and SDK identities. A changed deployment requires
  full payload verification; deployment records do not certify device health.
- Run serially through the supervised entry points with a fixed deadline. Bare native
  binaries do not provide the complete fault gate. Terminate a failed child and preserve
  its marker, logs and incomplete deployment before considering another attempt.
- Require numerical, execution and resource-release checks, then delayed kernel
  observation. Correct tokens alone do not establish successful acceptance.

## Fault handling

| Observation | Required action |
|---|---|
| NP hang (accelerator execution stall) | Stop the run; assess the bounded SoC retry conditions below. |
| EDMA (direct-memory-access) timeout, kernel BUG/Oops, page fault, watchdog or driver stop/dump | Stop board work and investigate; these are not isolated-hang retry cases. |
| SDHCI (SD-card host controller) timeout, EXT4 filesystem error or block IO error | Stop board work, preserve incomplete deployments and establish storage integrity before resuming. |
| Stuck process, remaining device owner or unreleased memory | Stop board work; do not launch another process over unresolved resources. |
| Software or numerical failure without a device fault | Preserve the failed result and marker; correct the specific issue and use an explicitly scoped diagnostic retry. |

`Waiting for HDMA Transfer...` alone is not a timeout. With driver fault dumping enabled,
`GPU DRV CONFIG` and `Stop driver to keep scene` are fault signatures. Do not ignore an
entire error class: an accepted baseline may acknowledge only exact recorded occurrences.

## Deadlines

| Scope | Preparation | Execution | Post-exit observation |
|---|---:|---:|---:|
| Formal SoC OPT | ≤240 s | ≤300 s | ≥10 s |
| Formal SoC Qwen | ≤240 s | Default 300 s; explicitly selected limit up to 600 s | ≥10 s |
| EP / unphased diagnostics | Combined child lifetime ≤300 s | Included in combined limit | EP ≥45 s; SoC ≥10 s |

The readiness signal starts the execution budget once for the entire workload; requests
cannot restart it. Never extend a running deadline. Deployment verification is separate
from model preparation. After timeout, terminate and preserve evidence before retrying.

The shorter SoC observation can miss a previously observed recovery Oops about 33 s after
exit. Read the complete kernel again before the next run, even after a clean observation.

## Retry and recovery

SoC may retry an isolated NP hang only after the failed process exits, device owners and
memory allocations are released, and delayed observation finds no further fault. Preserve
all incident markers. This exception neither clears the recovery gate nor applies to EP,
storage errors, other kernel faults or stuck processes.

Stop after three consecutive hang failures of the same experiment, counting the first.
Renaming or shortening it does not reset the count. Only complete acceptance resets the
consecutive failure count; it never erases incident history. Investigate offline after the limit.

Clearing a gate requires confirmed board power cycling, agreed recovery controls, numerical
and execution acceptance, resource release and the delayed checks above. A host cold restart
is not required, and a changed host boot ID or a soft restart alone is not proof of recovery.
Explicit diagnostic exceptions apply only to their recorded scope and do not clear old gates.
Archive markers only after all recovery conditions pass. If local evidence is missing, do not
infer a clean device or invent an exception; establish a new recovery baseline with the operator.

Do not restore retired convolution controls or silently substitute vendor convolution or
frozen model tests. Both have failed on confirmed cold boots. Replacement controls need
operator agreement; an EP control that fails after recovery needs new analysis before retrying.
Intentional hang reproduction requires explicit authorization and a board power-cycle reminder.

## Incident evidence

Keep local incident records and raw logs outside published source, separately from runtime
markers. Record each event's boot and artifact identities, input/graph configuration, last
completed and outstanding SDK calls, kernel timestamps, exit/release checks, recovery decision
and evidence locations. Distinguish observed facts from hypotheses. A later passing run does
not turn an earlier failure into acceptance.

Detailed local records are not a runtime dependency. The runner enforces machine-readable
markers and health checks, not the existence of a Markdown file. Maintain published findings
in [SDK and driver findings](np101-sdk-driver-validation-report.md), without per-run diaries.
