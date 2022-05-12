"""
This file creates the ClinVar gene variants dataset using a pretrained model, Enformer.
This file needs the 'variant_summary.txt' file from official ClinVar website as well as 'hg19.fa' and 'hg38.fa' in order to run.
"""
#%%
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tqdm import tqdm
import os
from GTDA.fasta_extractor import *
# Make sure the GPU is enabled 
assert tf.config.list_physical_devices('GPU'), 'Start the colab kernel with GPU: Runtime -> Change runtime type -> GPU'
# %%
class Enformer:

  def __init__(self, tfhub_url):
    self._model = hub.load(tfhub_url).model

  def predict_on_batch(self, inputs):
    predictions = self._model.predict_on_batch(inputs)
    return {k: v.numpy() for k, v in predictions.items()}

  @tf.function
  def contribution_input_grad(self, input_sequence,
                              target_mask, output_head='human'):
    input_sequence = input_sequence[tf.newaxis]

    target_mask_mass = tf.reduce_sum(target_mask)
    with tf.GradientTape() as tape:
      tape.watch(input_sequence)
      prediction = tf.reduce_sum(
          target_mask[tf.newaxis] *
          self._model.predict_on_batch(input_sequence)[output_head]) / target_mask_mass

    input_grad = tape.gradient(prediction, input_sequence) * input_sequence
    input_grad = tf.squeeze(input_grad, axis=0)
    return tf.reduce_sum(input_grad, axis=-1)
# %%
root = 'dataset'
dataset = 'variants'
selected_gene = 'BRCA1'
df = pd.read_table(f"{root}/{dataset}/variant_summary.txt",sep='\t')
model_save_dir = f"{root}/{dataset}/{selected_gene}"
if os.path.exists(model_save_dir) == False:
    os.makedirs(model_save_dir)
# %%
df = df[df['GeneSymbol'] == selected_gene][df['PositionVCF'] != -1].reset_index()
# %%
model_path = 'https://tfhub.dev/deepmind/enformer/1'
SEQUENCE_LENGTH = 393216
CENTER = 114688//128//2
model = Enformer(model_path)
fasta_file = f"{root}/{dataset}/hg19.fa"
fasta_extractor_37 = FastaStringExtractor(fasta_file)
fasta_file = f"{root}/{dataset}/hg38.fa"
fasta_extractor_38 = FastaStringExtractor(fasta_file)
# %%
all_ref_preds = []
all_alt_preds = []
valid_ids = []
for idx in tqdm(range(df.shape[0])):
    try:
        assembly = df.iloc[idx]['Assembly']
        chromosome = df.iloc[idx]['Chromosome']
        start_pos = df.iloc[idx]['Start']
        stop_pos = df.iloc[idx]['Stop']
        var_type = df.iloc[idx]['Type']
        vcf_pos = df.iloc[idx]['PositionVCF']
        # print(var_type)
        ref = df.iloc[idx]['ReferenceAlleleVCF']
        alt = df.iloc[idx]['AlternateAlleleVCF']
        # target_interval = kipoiseq.Interval(f'chr{chromosome}', start_pos-1, stop_pos)
        if assembly == "GRCh37":
            fasta_extractor = fasta_extractor_37
        elif assembly == "GRCh38":
            fasta_extractor = fasta_extractor_38
        # print(fasta_extractor.extract(target_interval))
        variant = kipoiseq.Variant(
            chrom=f'chr{chromosome}',
            pos=vcf_pos,
            ref=ref,
            alt=alt
        )
        interval = kipoiseq.Interval(variant.chrom, variant.start, variant.start).resize(SEQUENCE_LENGTH)
        seq_extractor = kipoiseq.extractors.VariantSeqExtractor(reference_sequence=fasta_extractor)
        center = interval.center() - interval.start
        reference = seq_extractor.extract(interval, [], anchor=center)
        alternate = seq_extractor.extract(interval, [variant], anchor=center)
        # Make predictions for the refernece and alternate allele
        reference_prediction = model.predict_on_batch(one_hot_encode(reference)[np.newaxis])['human'][0][(CENTER-2):(CENTER+2),:]
        alternate_prediction = model.predict_on_batch(one_hot_encode(alternate)[np.newaxis])['human'][0][(CENTER-2):(CENTER+2),:]
        all_ref_preds.append(reference_prediction.tolist())
        all_alt_preds.append(alternate_prediction.tolist())
        valid_ids.append(idx)
    except:
        continue
#%%
all_ref_preds = np.array(all_ref_preds)
all_alt_preds = np.array(all_alt_preds)
df = df.iloc[valid_ids].reset_index()
labels = df['ClinSigSimple'].values
labels[np.nonzero(labels==-1)] = 0
np.save(f"{model_save_dir}/reference_predictions.npy",all_ref_preds)
np.save(f"{model_save_dir}/alternative_predictions.npy",all_alt_preds)
np.save(f"{model_save_dir}/labels.npy",labels)
with open(f"{model_save_dir}/selected_ids.txt","w") as f:
    for i in valid_ids:
        f.write(f"{i}\n")

hg19_records = df[df['Assembly']=='GRCh37']
new_pos = pd.read_table(f"{model_save_dir}/hg19_to_hg38.bed",names=["chr","start","end"])["start"].values
hg19_to_hg38 = {}
for i in range(hg19_records.shape[0]):
    old_pos = hg19_records.iloc[i]['PositionVCF']
    hg19_to_hg38[old_pos] = new_pos[i]
vcf_pos = []
for i in range(df.shape[0]):
    if df.iloc[i]['Assembly'] == 'GRCh38':
        vcf_pos.append(df.iloc[i]['PositionVCF'])
    else:
        vcf_pos.append(hg19_to_hg38[df.iloc[i]['PositionVCF']])
vcf_pos = np.array(vcf_pos)
np.save(f"{model_save_dir}/vcf_pos.npy",vcf_pos)
all_significance = df['ClinicalSignificance'].values
with open(f"{model_save_dir}/all_significance.txt","w") as f:
    for i in all_significance:
        f.write(f"{i}\n")