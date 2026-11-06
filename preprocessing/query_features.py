"""Generate operon, promoter and topology arrays from CDSs and inferred operons."""
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from .genome_features import dna_onehot, scan_sigma_like_features


def build_features(genome, records, operons, output, unknown_topology='circular'):
    output = Path(output)
    genes = genome['genes']; count = len(genes)
    by_id = {g['gene_id']: g for g in genes}
    record_map = {r.id: r for r in records}
    gene_to_op = {}
    for i, op in enumerate(operons):
        for gid in op['genes']:
            if gid in gene_to_op:
                raise ValueError(f'Gene assigned to multiple operons: {gid}')
            gene_to_op[gid] = i
    if set(gene_to_op) != set(by_id):
        raise ValueError('Operon membership must cover every CDS exactly once')
    order = np.array([gene_to_op[g['gene_id']] for g in genes], dtype=np.int64)
    nops = len(operons)
    first = np.array([np.flatnonzero(order == i)[0] for i in range(nops)], dtype=np.int64)
    sizes = np.bincount(order, minlength=nops).astype(np.int64)
    strand = np.array([g['strand'] for g in genes], dtype=np.int64)
    inter = np.zeros(count, dtype=np.float32)
    for i, g in enumerate(genes):
        inter[i] = 1000.0 if i == 0 or g['contig_id'] != genes[i-1]['contig_id'] else max(0, g['start'] - genes[i-1]['end'])
    onehot = np.zeros((nops, 300, 5), dtype=np.float32)
    numeric = np.zeros((nops, 12), dtype=np.float32)
    promoters, promoter_by_gene = [], {}
    for i, op in enumerate(operons):
        members = [by_id[gid] for gid in op['genes']]
        # Match the original promoter recipe: 250 upstream + 50 downstream,
        # clipped at contig ends, no circular wrapping or intergenic truncation.
        fivep = min(members, key=lambda g: g['start']) if op['strand'] == 1 else max(members, key=lambda g: g['end'])
        rec = record_map[op['contig']]
        if op['strand'] == 1:
            begin, end = fivep['start'] - 250, fivep['start'] + 50
        else:
            begin, end = fivep['end'] - 50, fivep['end'] + 250
        begin, end = max(0, begin), min(len(rec.seq), end)
        seq = rec.seq[begin:end]
        if op['strand'] == -1:
            seq = seq.reverse_complement()
        sequence = str(seq).upper() if end - begin >= 30 else 'N' * 300
        onehot[i] = dna_onehot(sequence)
        numeric[i] = scan_sigma_like_features(sequence)
        promoters.append(SeqRecord(Seq(sequence), id=f'{op["operon_id"]}|loc={begin}-{end}|strand={op["strand"]}', description=''))
        for gid in op['genes']:
            promoter_by_gene[gid] = sequence
    for g in genes:
        g['promoter_sequence'] = promoter_by_gene[g['gene_id']]
    global_features = np.array([np.log1p(count), np.log1p(nops), inter.mean(), inter.std(),
        (strand > 0).mean(), (strand < 0).mean(), sizes.mean(), sizes.std(),
        numeric[:,0].mean(), numeric[:,2].mean(), numeric[:,3].mean(), numeric[:,4].mean()], dtype=np.float32)
    np.savez_compressed(output/'operon_meta.npz', operon_id_of_gene=order, gene_strand=strand,
        intergenic_upstream=inter, gene_role_type=np.zeros(count, np.int64), promoter_onehot=onehot,
        promoter_numeric_features=numeric, operon_valid_mask=np.ones(nops, bool),
        global_features=global_features, operon_sizes=sizes, operon_first_gene_idx=first)
    starts = np.zeros(nops, np.int32); lengths = np.zeros(nops, np.int32); topology = np.zeros(nops, np.int8)
    assumptions, cursor = [], 0
    while cursor < nops:
        contig = operons[cursor]['contig']; end = cursor + 1
        while end < nops and operons[end]['contig'] == contig:
            end += 1
        value = record_map[contig].annotations.get('topology', '').lower()
        if value not in ['linear', 'circular']:
            value = unknown_topology; assumptions.append(contig)
        starts[cursor:end] = cursor; lengths[cursor:end] = end-cursor
        topology[cursor:end] = int(value == 'circular'); cursor = end
    np.savez_compressed(output/'topology_meta.npz', operon_contig_start=starts,
                         operon_contig_len=lengths, operon_topology=topology)
    SeqIO.write(promoters, output/'promoters.fasta', 'fasta')
    pd.DataFrame([{**op, 'genes': ','.join(op['genes'])} for op in operons]).to_csv(output/'operons.tsv', sep='\t', index=False)
    return {'genes': count, 'operons': nops, 'unknown_topology_contigs': assumptions,
            'unknown_topology_default': unknown_topology, 'promoter_window': [250, 50]}
