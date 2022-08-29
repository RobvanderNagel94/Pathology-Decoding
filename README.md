# Detecting pathological EEGs using features of the background pattern

Electroencephalography (EEG) is a non-invasive technique used for
monitoring and recording brain electrical activity. EEGs provide
essential information about brain functioning and assist clinicians
in studying the long-term effects of neurological conditions like
epilepsy, Alzheimer’s disease, Parkinson’s disease, schizophrenia
and depression. Routine recordings are usually examined by board-certified clinicians that analyse recordings following two guidelines: recognition of transients and analysis of the background pattern. Transients refer to relatively
rare events, including physiological and pathological waveforms,
while the background pattern refers to the mean statistical characteristics of the EEG. Important components of the background pattern
include the posterior dominant rhythm (PDR), reactivity, frequency
distribution over the scalp, and the presence or absence of asymmetries. The presence of pathological transients, absent or abnormal
PDR peak frequencies, decreased reactivity and asymmetries across
hemispheres are all indicators of general brain dysfunction. 
The decision between a normal or pathological recording is made based
on a range of specific descriptions of the background pattern and
the presence or absence of transients, given the patient’s state of
consciousness (awake, asleep, drowsy, comatose), state (eyes-open,
eyes-closed, hyperventilation, photic stimulation) and medication
usage. 

Although visual analysis remains the gold standard for EEG
interpretation, computer-based algorithms have been proposed that
can detect abnormal electroencephalograms based on labeled EEG
recordings. Pathology decoding approaches are very promising and
offer major advantages to clinicians. While significant progress has been made in automated seizure
and spike detection through deep learning approaches, pathology
decoding research is still in its infancy, as satisfactory performance
levels have not yet been reached, and decoding approaches are far from replacing visual analysis.


# Contribution 

The literature on pathology decoding proposes a wide range of
feature-based and end-to-end approaches. However, little attention
is given to features that capture specific properties of the EEG
background activity, while changes in the background pattern can
be excellent indicators of brain dysfunction. Other than the papers
that focused on spectograms to capture the time-localized frequency
characteristics of the PDR, no other background properties were utilized. However, Lodder and van Putten (2012) proposed to quantify the properties of the background activity using five key features; alpha rhythm
frequency, reactivity, anterio–posterior gradients, asymmetries, and
diffuse slow-wave activity. The features assist in enhancing the consistency and reliability of inter-rater reporting of EEG background
activity, as well as enabling automated analysis. The features center
on the principle of analysing the spectral power measured along the
scalp. Build from these same principles, van Putten (2007) earlier
introduced to visualise the time-localised frequency information for
a triplet of features that quantifies the spatial distribution of the EEG
signals and their coherence, represented as three time-frequency
images (Colorful Brain). Features that describe the properties of
the background pattern might enhance decoding performance, as
these properties influence the decisions made to assign labels to
EEG recordings. Another strong motivator for using such features is
that they are already being used in clinical practice, enhancing feature interpretability and explainability. 

Shirrmeister et al. (2017) and van Leeuwen et al. (2019) have
set strong baselines to decode pathology using raw input data in
an end-to-end way. The approaches are essentially similar, but differ in minor degree. Therefore, the deep ConvNet model used in these
papers is set as baseline for comparison. We propose to build multiple
feature-based models; one class of models based on multichannel
data using a trial-wise decoding strategy, and the other class on
single-channel data using an image-based decoding strategy. In this
way we can systematically evaluate different decoding steps and
better assess the parts on their contribution to the total decoding
performance. 
