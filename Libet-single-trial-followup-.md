*. load test epochs (correctly and misclassified ones) for each classifier
*. identify which misclassified epochs overlap between which classifiers
*. construct an ensemble guided by an uncertainty measure and test if the ensemble improves upon individual classifiers 
*. graph of intermediate probabilities in a classifier's computation

*. SPD matrices (nodes), FgMDM of SPD matrices (edges) ~> persistence diagram of this graph over [,]s 
    *. [,] = computation time of barcodes
*. notion of a path signature of [phi_1,...,phi_9] of EEG time series over target epochs vs resting (nontarget) epochs 

*. how does Xiaoqi Xu apply TDA to EEG streams: who cares.. use her example and use my own ingenious way  

*. test spline framework for the surface Laplacian 
*. test vectorization methods (that are intuitive) for TDA so we can classify over Euclidean space: proof is in the pudding so skip the descriptive bullshit and go straight for the jugular i.e does it work empirically ?

*. debrief of subjects
*. introduce movement inhibition task in between cues for self paced motor actions: can EEG discriminate between pressing a button with a finger from pressing a pedal with a foot ? 128 channels, source reconstruction..
*. strong vs weak intention to move driven by the task
*. crap classification accuracy on Stuart's data
*. MNE-C, MNE-LSL 

*. controlled LLM task generation to elicit EEG patterns 

*. prep_childmind_eeg_batch-gpu.py contains EEG matrix computed on CUDA