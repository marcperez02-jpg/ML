```
BUILDING AN AVIAN INFLUENZA H5N1 CLADE 2.3.4.4b PREDICTOR WITH MACHINE
LEARNING
```
**INTRODUCTION**

Avian influenza (AI) represents a significant poultry disease resulting from infection with type A influenza viruses, commonly designated as avian influenza viruses (AIV) (Swayne and Spackman 2013). These viruses are characterized by a genome comprising eight distinct gene segments and are taxonomically classified into various subtypes according to the specific composition of their surface glycoproteins. This classification system relies on the 16 hemagglutinin (HA) subtypes (H1 through H16) and nine neuraminidase (NA) subtypes (N1 through N9), which are encoded by gene segments 4 and 6 (Swayne and Sims, 2021).

Poultry-infecting influenza A viruses are categorized into two distinct groups based on their clinical impact. Highly pathogenic avian influenza (HPAI) represents the most severe form, capable of causing up to 100% mortality within a flock (Hurt et al., 2017). While this high-virulence phenotype is restricted to the H5 and H7 subtypes, it is important to note that not all viruses within these two subtypes are highly pathogenic. Most other strains result in low pathogenic avian influenza (LPAI), which typically manifests as a mild respiratory condition unless secondary factors increase the severity of the disease (Alexander, 2007).

The global proliferation of HPAI H5N1 since 2021 has developed into an unprecedented panzootic, resulting in catastrophic poultry losses and frequent spillover events into diverse mammalian hosts (Harvey et al., 2023). This expansion has naturally increased human exposure risks, thereby heightening the potential for a future pandemic (Krammer et al., 2025). Due to the rapid accumulation of novel mutations, traditional subtype classifications no longer fully represent the virus's complex evolutionary history. To address this limitation, the WHO/WOAH/FAO H5 working group established a specialized clade-based classification system to more accurately track these genetic shifts (WHO/OIE/FAO, 2014).

Recent human infections involving clade 2.3.4.4b, acquired from contact with infected poultry or dairy cattle, have demonstrated a broad clinical spectrum ranging from mild conjunctivitis to fatal pneumonia. Consequently, there is an urgent need for robust genomic surveillance and rapid data sharing across avian, mammalian, and human populations to monitor the ongoing evolution of these viruses (Webby and Uyeki, 2024).  As emphasized by Graziosi et al. (2024), the dominance of the 2.3.4.4b lineage necessitates precise pathogen characterization. Effective public health responses depend entirely on the identification of viral clades, a task made challenging by the intricate and fluid evolutionary dynamics inherent to influenza A viruses.
The objective of this project is to develop a machine learning-based diagnostic tool capable of identifying whether a HA sequence from an influenza A virus belongs to the H5N1 clade 2.3.4.4b. 



 

**METHODS**

This GitHub repository (https://github.com/marcperez02-jpg/ML.git) contains a machine
learning pipeline that helps identify if a HA sequence of an AI belongs to the 2.3.4.4b clade.
Using a Snakefile, the workflow automates data processing and builds a predictive model.

To train the model on what to look for, the pipeline uses two distinct groups of sequences
pulled from the GISAID database in FASTA format. The target group, or positive dataset,
consists of a collection of sequences from clade 2.3.4.4b. In contrast, the comparison group,
or negative dataset, contains a broad mix of other H5 sequences that belong to any branch
except the 2.3.4.4b group.

The _build_test_set.py_ script serves as the initial data processing stage, utilizing the
_SeqIO.parse_ command from the Biopython library to read the _pos_records_ (positive records)
and _neg_records_ (negative records) from their respective FASTA files, previously mentioned.
The script first verifies that both lists contain at least 50 entries, ensuring the pipeline has a
sufficient volume of genomic data to proceed. To maintain reproducibility, it initializes a
_random.Random_ object with a fixed seed of 24, which guarantees that the _rng.sample_
function picks the exact same 50 sequences for the _pos_sample_ and _neg_sample_ variables
every time the code is executed with this seed.

Once the subsets are selected, the script enters a loop to modify the _record.description_
attribute for each sequence. It uses _Python f-strings_ to append _label=1_ to the target clade
sequences and _label=0_ to the control sequences, creating a reference within the FASTA
headers. Following this labeling, the _os.makedirs_ command ensures the data/test directory
exists before the _SeqIO.write_ command is called. This final step concatenates the two
samples and saves them into a single _test.fasta_ file, establishing a balanced, 100-sequence
dataset ready for unbiased model evaluation.

The _data_cleaning.py_ script manages the cleaning, balancing, and filtering of the training
data to ensure the model learns from a high-quality dataset. The workflow begins with the
_process_fasta_ function, which uses _SeqIO.parse_ from the Biopython library to load
sequences into memory. During this phase, it accesses the _record.seq_ attribute for each
entry, converting the biological sequence object into a standard Python string with
_str(record.seq).upper()._ This transformation is essential because it allows the script to
perform string-based operations, such as checking if the length falls between 1600 and 1900
nucleotides (The HA segment usually has 1700-1800 nucleotides) and ensuring that "N"
bases (unknown nucleotides) do not exceed 1% of the sequence.

Once the raw data is stored in the _df_raw_ Pandas DataFrame, the script immediately calls
_filter_overlapping_sequences_. By loading _test.fasta_ and using the command
_df_raw[~df_raw["sequence"].isin(test_sequences)]_ at this stage, the script erases any
sequence used for testing from the training pool. By filtering the raw data before counting the
classes, the variables _pos_count_ and _neg_count_ reflect only the sequences truly available
for training. This ensures the model remains strictly independent and doesn't “cheat” by
seeing test data during its training phase.
After the exclusion is complete, the script handles class imbalance by identifying the
_minority_count_ between the positive and negative samples. It creates two new variables,


 

_df_positive_balanced_ and _df_negative_balanced_ , by using the _.sample(n=minority_count)_
method with a fixed _random_state_ of 24, again for reproducibility. This downsamples the
majority class so that both groups are identical in size. Finally, it uses _pd.concat_ to merge
these groups and _sample(frac=1)_ to shuffle the entire dataset before saving it as
_balanced_dataset.csv._ This provides a randomized and balanced file for the training phase.

The _train_model.py_ script serves as the computational phase where genomic information is
converted into a mathematical format to train a predictive model. It begins by loading the
_balanced_dataset.csv_ , previously created, using Pandas and preparing for a technique
known as k-mer featurization (Kwon, 2021 ; Roberts et al., 2025). Because machine learning
models are unable to process raw text strings directly, the script employs the
_get_kmer_frequencies_ function to decompose each sequence into three-nucleotide
segments. It generates all 64 possible combinations of the bases A, C, G, and T, then counts
the occurrences of each in a sequence. Finally, it turns those counts into percentages. This
step, called normalization.

The script utilizes the _RandomForestClassifier_ from the _sklearn.ensemble_ module. The term
ensemble refers to a machine learning strategy that combines the predictions of multiple
individual models to create a more accurate and stable result than any single model could
achieve alone (Murel and Kavlakoglu, n.d). A Random Forest specifically builds a large
collection of independent decision trees, where each tree is trained on a random subset of
the data and the final prediction is determined by a majority vote among all trees (Breiman,
2001). This technique was chosen because it can capture complex, non-linear relationships
between different parts of the DNA sequence. It is also highly robust, meaning it is less likely
to overfit or memorize specific data points compared to a single decision tree. Furthermore,
it provides transparency by allowing to eventually identify which specific 3-mers were most
influential in identifying the 2.3.4.4b clade, using the .feature_importances method.

The script follows a structured workflow to move from raw data to a serialized model file. The
command _x = np.array([get_kmer_frequencies(seq) for seq in df["sequence"]])_ creates the
feature matrix by applying the k-mer function to every sequence in the dataframe, while _y =
df["label"].values_ extracts the labels previously created (0 = negative, 1 = positive). To
initialize the model, the command _RandomForestClassifier(n_estimators=100,
random_state=24)_ is used to specify that the forest should consist of 100 individual trees.

The actual learning process occurs via the _model.fit(x, y)_ command, where the model
analyzes the patterns in the k-mer frequencies and learns how they correlate with the
assigned labels. Finally, the script uses the _joblib_ library, specifically the _joblib.dump_
command, to save the trained model. This library is used instead of standard Python saving
methods because it is highly optimized for handling the large NumPy arrays and complex
structures found in scikit-learn models (Joblib, n.d). This process saves the trained predictor
model as a .pkl file in the models directory, allowing it to be reloaded later for testing or
real-world classification without the need for retraining.

The validate_predictor.py is designed to assess the predictive performance of the trained
Random Forest model on an independent test set previously created.


 

It functions by using the _joblib.load_ command to bring the model back into memory and then
processes each test sequence through the _get_kmer_frequencies_ function, explained above.
Throughout the execution, the script compares the model’s predictions against the labels
extracted by the _get_label_from_description_ function. This comparison allows for the
calculation of statistical metrics using _precision_score_ and _recall_score_ , which define the
model's reliability and its ability to detect all target sequences without missing high-risk
variants. The _confusion_matrix_ command is also utilized, categorizing every prediction into a
scorecard of true positives, true negatives, false positives, or false negatives. This methods
are extracted from the library _sklearn.metrics_ (scikit-learn, n.d.) and are stored in
_models/validation_report.txt_.

Beyond those metrics, the script extracts the _model.feature_importances__ attribute to
identify which specific 3-mer motifs were most influential in the decision-making process.

The visualize.py script turns the raw numbers from the validation report into graphs. It uses
the _matplotlib.pyplot_ library to generate three distinct types of visuals that make the model's
performance easy to interpret. Specifically, it generates a bar chart for precision and recall
and a heatmap for the confusion_matrix. By reloading the model with _joblib.load,_ it accesses
model.feature_importances_ to identify the ten most significant genetic 3-mers used for
classification. These results are saved as PNG files in the images folder.

The entire pipeline is managed by a Snakefile, which automates the workflow. Executing the
command _snakemake --cores 1_ triggers the full sequence of data cleaning, training, and
evaluation in the correct order. Additionally, a clean rule, which resets the project by
removing intermediate files and generated images, was implemented with the command
_snakemake clean --cores 1_.


 

# RESULTS

**Model Performance**

The evaluation of the Random Forest classifier on the independent test set yielded a
Precision of 1.00 and a Recall of 1.00 ( _Figure 1_ ). The confusion matrix confirms these
metrics, showing that of the 100 sequences tested (50 from clade 2.3.4.4b and 50 from other
clades), all were classified with 100% accuracy ( _Figure 2_ ). Specifically, there were 50 True
Positives and 50 True Negatives, with zero instances of False Positives or False Negatives.

<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/9a14564c-de8d-4ddf-9ba3-b4edba1d69a0" />

```
Figure 1. Model Performance Scores. This graph shows that the model achieved a perfect score of
1.0 (100%) for both Precision and Recall.
```
<img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/45df91e1-0ca4-49b2-a478-d28d129032a1" />

```
Figure 2. Confusion Matrix. This heatmap provides a detailed breakdown of the model's predictions.
The dark blue squares show that all 50 "Other" sequences and all 50 "2.3.4.4b" sequences were
classified correctly.
```

 

**Feature Importance**

The analysis of the top 10 most influential 3-mers revealed that the GCG and CCG motifs
were the primary drivers for clade differentiation, possessing significantly higher importance
scores (approx. 0.18 and 0.125, respectively) than the remaining features ( _Figure 3_ ). These
motifs likely represent conserved codons or regulatory regions specific to the HA segment of
the 2.3.4.4b lineage. The high importance of specific 3-mers suggests that the Random
Forest has successfully isolated conserved genomic motifs that serve as definitive molecular
signatures for this H5N1 lineage

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/b3b45939-f830-41e7-9034-81a90558ba3a" />

```
Figure 3. Relative Feature Importance of Top 10 Influential 3-mers. This visualization
quantifies the contribution of specific trinucleotide motifs to the Random Forest model's
```
## classification logic.

### DISCUSSION

Achieving “perfect” scores of 1.00 raises concerns about overfitting or potential data
leakage. In machine learning, a model is considered overfitted when it memorizes noise or
specific characteristics of the training data rather than learning generalizable biological
patterns, which typically leads to poor performance on unseen data (IBM ,n.d.). Although
strict precautions were implemented to prevent data leakage, the possibility of it happening
cannot be fully excluded.


 

The pipeline incorporated a _filter_overlapping_sequences_ function that explicitly removed
any sequence present in the test set from the training dataset before model construction.
This step ensured that the model did not directly encounter test examples during training.

From a biological perspective, the strong separability observed in the results is scientifically
possible. Clade 2.3.4.4b follows distinct evolutionary trajectories and contains characteristic
mutations in the HA gene. This subclade involves the most recent prevalent strains with
multiple H5Nx subtypes, is globally spread, and forms the largest monophyletic subclade
(Demirev et al., 2023). Viral genomic clades often exhibit conserved sequence signatures
that differ substantially from other lineages. Consequently, a Random Forest classifier
trained on 3-mer frequencies may be capable of detecting these consistent molecular
patterns with high accuracy.

The dataset was also subjected to strict preprocessing. Only full-length HA sequences
between 1600 and 1900 nucleotides were retained, and sequences with more than 1%
ambiguous bases were excluded. This high-quality filtering reduces noise and prevents
distorted k-mer distributions, enabling the model to learn clear sequence patterns.

However, despite these methodological controls, perfect classification performance may
still indicate overfitting to dataset-specific characteristics. If the sequences within each class
share similar geographic origins, collection periods, or evolutionary backgrounds, the model
may be capturing subtle distributional biases rather than universally generalizable
clade-defining features.

Therefore, while the results are biologically possible and not necessarily the consequence
of data leakage, additional validation on distinct datasets would be necessary to confidently
exclude overfitting and confirm the model’s true predictive capacity.


 

**REFERENCES**

Alexander, D. J. (2007). An overview of the epidemiology of avian influenza. Vaccine, 25(30), 5637–5644. https://doi.org/10.1016/j.vaccine.2006.10.051
Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324 
Demirev, A. V., Park, H., Lee, K., Park, S., Bae, J.-Y., Park, M.-S., & Kim, J. I. (2023). Phylodynamics and Molecular Mutations of the Hemagglutinin Affecting Global Transmission and Host Adaptation of H5Nx Viruses. Transboundary and Emerging Diseases, 2023, 1–14. https://doi.org/10.1155/2023/8855164
Graziosi, G., Lupini, C., Catelli, E., & Carnaccini, S. (2024). Highly Pathogenic Avian Influenza (HPAI) H5 Clade 2.3.4.4b Virus Infection in Birds and Mammals. Animals, 14(9), 1372. https://doi.org/10.3390/ani14091372
Harvey, J. A., Mullinax, J. M., Runge, M. C., & Prosser, D. J. (2023). The changing dynamics of highly pathogenic avian influenza H5N1: Next steps for management & science in North America. Biological Conservation, 282, 110041. https://doi.org/10.1016/j.biocon.2023.110041
Hurt, A. C., Fouchier, R. A. M., & Vijaykrishna, D. (2017). Ecology and Evolution of Avian Influenza Viruses. In Genetics and Evolution of Infectious Diseases (pp. 621–640). Elsevier. https://doi.org/10.1016/B978-0-12-799942-5.00027-5
IBM. (n.d.). What is overfitting? IBM. 
https://www.ibm.com/think/topics/overfitting 
Joblib. (n.d.). In *Joblib documentation*. Read the Docs. https://joblib.readthedocs.io/en/stable/ 
Krammer, F., Hermann, E., & Rasmussen, A. L. (2025). Highly pathogenic avian influenza H5N1: History, current situation, and outlook. Journal of Virology, 99(4), e02209-24. https://doi.org/10.1128/jvi.02209-24
Kwon, B. (2021, June 28). Machine learning for biological sequence data using Python. Medium. https://byeungchun.medium.com/machine-learning-for-biological-sequence-data-using-python-573d82f6f17a 
Murel, J., & Kavlakoglu, E. (n.d.). What is ensemble learning? IBM. https://www.ibm.com/think/topics/ensemble-learning 
Roberts, M. D., Davis, O., Josephs, E. B., & Williamson, R. J. (2025). K -mer-based Approaches to Bridging Pangenomics and Population Genetics. Molecular Biology and Evolution, 42(3), msaf047. https://doi.org/10.1093/molbev/msaf047
scikit-learn. (n.d.). *sklearn.metrics — scikit-learn API documentation*. https://scikit-learn.org/stable/api/sklearn.metrics.html 
Swayne, D. E., & Sims, L. D. (2021). Avian Influenza. In S. Metwally, A. El Idrissi, & G. Viljoen (Eds), Veterinary Vaccines (1st edn, pp. 229–251). Wiley. https://doi.org/10.1002/9781119506287.ch18
Swayne, D. E., & Spackman, E. (2013). Current status and future needs in diagnostics and vaccines for high pathogenicity avian influenza. Developments in biologicals, 135, 79–94. https://doi.org/10.1159/000325276
Webby, R. J., & Uyeki, T. M. (2024). An Update on Highly Pathogenic Avian Influenza A(H5N1) Virus, Clade 2.3.4.4b. The Journal of Infectious Diseases, 230(3), 533–542. https://doi.org/10.1093/infdis/jiae379
World Health Organization/World Organisation for Animal Health/Food and Agriculture Organization (WHO/OIE/FAO) H5N1 Evolution Working Group. (2014). Revised and updated nomenclature for highly pathogenic avian influenza A (H5N1) viruses. Influenza and Other Respiratory Viruses, 8(3), 384–388. https://doi.org/10.1111/irv.12230

