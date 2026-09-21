# TransH branch implementation note

The TransH branch normalizes the relation-specific hyperplane normal before projection so that the projection is not affected by the raw normal-vector scale. The relation translation vector remains an independent trainable parameter. This definition is used consistently by the TH-RotatE family and its direct controls in the repository.

The RotatE branch uses the corresponding Euclidean-form residual distance before fusion. The manuscript's MPNorm experiment separately tests whether the comparative result can be explained by raw branch-distance scale differences.
