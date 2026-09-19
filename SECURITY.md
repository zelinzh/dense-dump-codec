# Security and data safety

DDC readers are for trusted local research archives. CRC/SHA checks detect
accidental corruption but do not authenticate the publisher. Manifests and
archive metadata reference external paths and can request large allocations;
do not serve untrusted archives or expose the Unix socket to untrusted users.
Checksums can be replaced together with files by an attacker.

Keep backups. Do not enable source deletion, checkpoint pruning or permission
repair without understanding the selected paths. Work in a dedicated output
directory and do not run concurrent writers there. Some legacy tools retry
permission failures by making owned artifacts readable; shared sensitive inputs
should not be used with those workflows without review.

Report suspected vulnerabilities privately to the repository maintainer using
the hosting platform's private reporting facility if enabled.
