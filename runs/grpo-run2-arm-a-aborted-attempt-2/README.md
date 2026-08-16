# Arm A, aborted attempt 2

Completed all 300 steps and all three checkpoint evaluations, then refused to
publish: the validator read `saved_checkpoints`, an attribute only the test
fake has. See tracker section 117.

`zero-variance-per-step.txt` holds the per-step `frac_reward_zero_std` values
for all 300 steps. 125 were 1.0. The completed run reproduced that figure
exactly, so this file is corroboration rather than unique evidence.
