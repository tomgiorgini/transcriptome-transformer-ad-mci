# TxT multitask con PPI Volumetric Attention

Questa directory contiene il protocollo riproducibile affiancato alla TxT multitask corrente. Il backbone denso, le tre teste class-only, il batch bilanciato e gli split restano invariati; la variante `ppi_volumetric` aggiunge soltanto un ramo sparse sugli archi HIPPIE, fuso da un gate residuale inizializzato a zero.

I launcher non vengono eseguiti automaticamente durante l'implementazione.

## Configurazione bloccata

Il confronto principale usa:

- split stratificati 70/10/20, seed 101–110;
- top-2000 per varianza, selezionati esclusivamente sul train;
- embedding random di dimensione 64;
- 1 layer, 2 head, `d_model=64`, FFN 256, dropout 0,2;
- batch 9 con `balanced_classes`, quindi 3 pazienti per classe biologica;
- encoder condiviso sui 9 pazienti e teste mask-aware su 6 righe valide per task;
- average pooling e pesi loss 0,5/0,25/0,25;
- massimo 200 epoche come limite di sicurezza, patience tradizionale disabilitata (`0`), LR `1e-4`, weight decay `1e-4` e arresto quando la validation loss resta sopra 1,0 per 5 epoche consecutive;
- checkpoint score `0.70*AUC(AD-MCI) + 0.15*AUC(AD-CTL) + 0.15*AUC(MCI-CTL) - 0.25*loss`;
- soglia HIPPIE inclusiva `score >= 0.73`, `epsilon=1e-8`, gate iniziale `gamma=0`;
- candidati beta `{0, 0.5, 1.0, 1.5}`.

`beta=0` è il controllo **PPI pair-wise**: il ramo sparse e i suoi gate query-dependent esistono ancora. Non è la baseline senza ramo.

## Modalità del volume e pairing

La modalità predefinita `raw` conserva esattamente la formula storica e, se
`--volumetric-dropout` non è specificato, eredita il dropout del backbone. È la
configurazione da usare per ricostruire i checkpoint già prodotti.

Per i nuovi pilot è disponibile `--volumetric-volume-mode l2`: normalizza L2
`q_i`, `k_i` e `k_j` prima del determinante, applica `-beta * volume` senza una
seconda divisione per `sqrt(d_head)` e mantiene il dot-product neighbor con lo
scaling standard. Le diagnostiche riportano anche media/deviazione standard del
volume ed entropia segmentata normalizzata.

Per un confronto paired più stretto si usano inoltre:

- `--volumetric-dropout 0`, che impedisce al ramo VMA di consumare numeri casuali
  aggiuntivi mentre il gate è ancora zero;
- `--post-model-construction-reseed on`, che riallinea lo stato RNG dopo la
  costruzione, diversa fra baseline e variante;
- `--grad-clip-scope separate_volumetric`, che non fa entrare il gradiente dei
  parametri VMA nella norma usata per tagliare i gradienti del backbone.

I default generici `raw`, dropout ereditato e clipping `joint` restano disponibili
per la compatibilità storica. Usare sempre una nuova `--result-root` quando si
cambia una di queste policy: `--skip-existing` non deve mescolare artifact di
protocolli differenti.

## Protocollo completo

Dal root del repository:

```bash
python3 experiments/scripts/paper_comparison/txt_volumetric/run_paired_protocol.py \
  --device cuda \
  --epochs 200 \
  --early-stopping-patience 0 \
  --max-genes 2000 \
  --ppi-edge-file pretraining_dataset/ppi_networks/hippie_highconf_edges.csv \
  --result-root results/paper_comparison/txt_volumetric
```

Per ogni seed, il launcher:

1. genera uno split stratificato deterministico 70/10/20;
2. addestra una baseline random-init e ne valuta il test;
3. addestra i quattro candidati VMA con `--evaluate-test off`;
4. sceglie beta leggendo esclusivamente il miglior score di validation salvato dal worker; una parità esatta sceglie il beta minore;
5. richiama il worker con `--evaluation-only-checkpoint` sul solo checkpoint selezionato;
6. calcola delta paired `ppi_volumetric - baseline` per seed/task/metrica, media, deviazione standard, bootstrap paired 95% deterministico e frequenze dei beta.

Il comando è riprendibile con `--skip-existing`. Le fasi possono essere lanciate separatamente con `--phase train`, `--phase select-evaluate` e `--phase summarize`. `--dry-run` stampa e salva i comandi di training senza iniziare run lunghi.

Durante il training ogni riga del worker, incluse le metriche di ciascuna epoca, viene mostrata nel terminale e contemporaneamente salvata nel relativo `worker.log`.

Sono disponibili anche entry point separati, utili su cluster o per una revisione esplicita del confine validation/test:

```bash
python3 experiments/scripts/paper_comparison/txt_volumetric/select_beta.py --result-root RESULTS
python3 experiments/scripts/paper_comparison/txt_volumetric/evaluate_selected.py --result-root RESULTS --device cuda
python3 experiments/scripts/paper_comparison/txt_volumetric/summarize_paired.py --result-root RESULTS
```

`select_beta.py` non apre file di predizione o metriche test. `evaluate_selected.py` riusa il comando originale salvato nel candidato e cambia soltanto result directory, device opzionale e modalità evaluation-only.

## Smoke CPU

```bash
python3 experiments/scripts/paper_comparison/txt_volumetric/run_paired_protocol.py \
  --smoke \
  --volumetric-volume-mode l2 \
  --volumetric-message-mode expression_contrast \
  --volumetric-output-norm rms \
  --volumetric-gate-mode per_head \
  --volumetric-backbone-gradient-mode detached \
  --volumetric-gate-init 0 \
  --volumetric-dropout 0 \
  --post-model-construction-reseed on \
  --grad-clip-scope separate_volumetric \
  --lr-volumetric 0.0005 \
  --checkpoint-ensemble-size 3 \
  --checkpoint-ensemble-min-gap 3 \
  --result-root results/paper_comparison/txt_volumetric_smoke
```

Lo smoke forza CPU, un seed, un'epoca, due batch train/validation, 64 geni e due run: baseline più VMA con `beta=1` (modificabile con `--smoke-beta`). Non esegue selezione, summary o valutazione test; il report fallisce anche se trova righe test inattese. La riduzione dei geni è una deroga esplicita per rendere il controllo realmente pratico su CPU; architettura, batch, mask e loss weights restano quelli fissati. Al termine viene scritto `smoke_artifact_report.json` e il comando fallisce se mancano artifact obbligatori.

## Pilot L2 paired su Windows

Da PowerShell, prima eseguire un solo seed e fermarsi alla validation dei
candidati (`--phase train`):

```powershell
Set-Location <repository-root>

$pilotRoot = "results\paper_comparison\txt_volumetric_l2_fair_pilot"

python experiments\scripts\paper_comparison\txt_volumetric\run_paired_protocol.py `
  --phase train `
  --device cuda `
  --seeds 101 `
  --betas 0 0.5 1 2 4 `
  --max-genes 2000 `
  --epochs 200 `
  --early-stopping-patience 0 `
  --volumetric-volume-mode l2 `
  --volumetric-dropout 0 `
  --post-model-construction-reseed on `
  --grad-clip-scope separate_volumetric `
  --ppi-edge-file pretraining_dataset\ppi_networks\hippie_highconf_edges.csv `
  --result-root $pilotRoot
```

Selezionare e leggere beta usando soltanto la validation, senza eseguire il
test del candidato:

```powershell
python experiments\scripts\paper_comparison\txt_volumetric\select_beta.py `
  --result-root $pilotRoot `
  --seeds 101 `
  --betas 0 0.5 1 2 4

Get-Content "$pilotRoot\seed_101\selected_beta.json"
```

Solo se scala, entropia e score di validation mostrano che il termine
volumetrico è diventato effettivo, estendere ai primi tre seed riusando il seed
101 già completo:

```powershell
python experiments\scripts\paper_comparison\txt_volumetric\run_paired_protocol.py `
  --phase train `
  --skip-existing `
  --device cuda `
  --seeds 101 102 103 `
  --betas 0 0.5 1 2 4 `
  --max-genes 2000 `
  --epochs 200 `
  --early-stopping-patience 0 `
  --volumetric-volume-mode l2 `
  --volumetric-dropout 0 `
  --post-model-construction-reseed on `
  --grad-clip-scope separate_volumetric `
  --ppi-edge-file pretraining_dataset\ppi_networks\hippie_highconf_edges.csv `
  --result-root $pilotRoot

python experiments\scripts\paper_comparison\txt_volumetric\select_beta.py `
  --result-root $pilotRoot `
  --seeds 101 102 103 `
  --betas 0 0.5 1 2 4

Import-Csv "$pilotRoot\beta_selections.csv" | Format-Table
```

Non lanciare ancora `--phase select-evaluate`: quella fase apre il test del
candidato selezionato e va eseguita soltanto dopo aver congelato la scelta.

## Pilot VMA v2 validation-only

Il profilo VMA v2 e' opt-in. I default del worker restano
`raw/legacy/none/scalar/coupled`, così i checkpoint storici continuano a usare
il percorso originale. VMA v2 usa invece:

- espressione standardizzata per paziente nelle proiezioni sparse Q/K/V;
- messaggio graph contrast `neighbor - self`;
- volume L2 adimensionale e normalizzazione RMS per singolo paziente;
- gate separato per head e gradiente VMA staccato dal backbone;
- dropout VMA zero, gate iniziale zero e learning rate VMA separato;
- ensemble top-3 come media aritmetica delle probabilita'.

Il controllo `beta=0` resta un modello con ramo PPI attivo e serve soltanto a
capire se il termine volumetrico aggiunge informazione. Non e' la baseline TxT
senza ramo. `beta=1` è fissata globalmente e non viene scelta separatamente per
seed. Dal root del repository, il pilot preregistrato e':

```powershell
$pilotRoot = "results\paper_comparison\txt_volumetric_v2_val_pilot"

python experiments\scripts\paper_comparison\txt_volumetric\run_paired_protocol.py `
  --phase train `
  --baseline-evaluate-test off `
  --device cuda `
  --seeds 201 202 203 `
  --betas 0 1 `
  --beta-policy global_fixed `
  --primary-beta 1 `
  --max-genes 2000 `
  --epochs 200 `
  --early-stopping-patience 0 `
  --volumetric-volume-mode l2 `
  --volumetric-message-mode expression_contrast `
  --volumetric-output-norm rms `
  --volumetric-gate-mode per_head `
  --volumetric-backbone-gradient-mode detached `
  --volumetric-gate-init 0 `
  --volumetric-dropout 0 `
  --post-model-construction-reseed on `
  --grad-clip-scope separate_volumetric `
  --lr 0.0001 `
  --lr-volumetric 0.0005 `
  --checkpoint-ensemble-size 3 `
  --checkpoint-ensemble-min-gap 3 `
  --ppi-edge-file pretraining_dataset\ppi_networks\hippie_highconf_edges.csv `
  --result-root $pilotRoot
```

Questo comando non deve produrre righe o prediction del test, neppure per la
baseline. L'audit meccanistico successivo e anch'esso vincolato alla validation:

```powershell
python experiments\scripts\paper_comparison\txt_volumetric\audit_validation.py `
  --result-root $pilotRoot `
  --seeds 201 202 203 `
  --primary-beta 1 `
  --control-beta 0 `
  --device cuda

Get-Content "$pilotRoot\validation_audit\validation_audit_summary.json"
Import-Csv "$pilotRoot\validation_audit\validation_audit_by_seed.csv" | Format-Table
```

L'audit confronta l'ensemble completo con `gate=0`, azzera beta soltanto in
inference e misura la total variation delle distribuzioni sparse sui target
PPI con almeno due vicini. Il pilot passa solo se tutti i controlli meccanici
sono attivi e lo score validation di `beta=1` supera in media sia baseline sia
il controllo addestrato `beta=0`, con segno positivo in almeno due seed su tre.

Se il pilot passa, l'espansione preregistrata usa soltanto `beta=1` globale:

```powershell
$tenSeedRoot = "results\paper_comparison\txt_volumetric_v2_val_10seed"

python experiments\scripts\paper_comparison\txt_volumetric\run_paired_protocol.py `
  --phase train `
  --baseline-evaluate-test off `
  --device cuda `
  --seeds 201 202 203 204 205 206 207 208 209 210 `
  --betas 1 `
  --beta-policy global_fixed `
  --primary-beta 1 `
  --max-genes 2000 `
  --epochs 200 `
  --early-stopping-patience 0 `
  --volumetric-volume-mode l2 `
  --volumetric-message-mode expression_contrast `
  --volumetric-output-norm rms `
  --volumetric-gate-mode per_head `
  --volumetric-backbone-gradient-mode detached `
  --volumetric-gate-init 0 `
  --volumetric-dropout 0 `
  --post-model-construction-reseed on `
  --grad-clip-scope separate_volumetric `
  --lr 0.0001 `
  --lr-volumetric 0.0005 `
  --checkpoint-ensemble-size 3 `
  --checkpoint-ensemble-min-gap 3 `
  --ppi-edge-file pretraining_dataset\ppi_networks\hippie_highconf_edges.csv `
  --result-root $tenSeedRoot

python experiments\scripts\paper_comparison\txt_volumetric\summarize_validation_expansion.py `
  --result-root $tenSeedRoot `
  --seeds 201 202 203 204 205 206 207 208 209 210 `
  --primary-beta 1
```

Il summarizer non espone un'opzione test e dà il via libera soltanto con delta
validation medio positivo e almeno 7 seed positivi su 10.

Non usare `--phase select-evaluate` su questa root durante il pilot. Se i
criteri passano, la configurazione primaria viene congelata a `beta=1` globale
e ripetuta validation-only sui seed 201-210 in una nuova result root. Soltanto
dopo il go/no-go sui dieci seed si esegue la fase finale, che valuta una volta
gli ensemble congelati di baseline e VMA in directory evaluation-only separate.

`--skip-existing` riusa un job soltanto quando fingerprint di comando, input,
split e PPI coincidono. Per le evaluation deve coincidere anche l'elenco degli
SHA-256 dei checkpoint. Un mismatch termina con errore: non viene rietichettata
o sovrascritta una run precedente.

## Artifact

Layout principale:

```text
RESULTS/
  protocol_manifest.json
  split_manifest.json
  splits/seed_101.csv
  seed_101/
    baseline/
    baseline_test/
    candidates/beta_0/
    candidates/beta_0p5/
    candidates/beta_1/
    candidates/beta_1p5/
    selected_beta.json
    selected_test/
  paired_deltas.csv
  paired_summary.csv
  beta_selections.csv
  beta_selection_frequencies.csv
  summary_manifest.json
```

Ogni directory VMA contiene `induced_ppi_edges.csv`, `ppi_graph_manifest.json`, `volumetric_diagnostics.json`, `training_log.csv`, `best_model.pt`, geni selezionati e artifact standard del worker. Il manifest del grafo conserva hash SHA-256 HIPPIE, threshold, coverage di nodi/archi, isolati e distribuzione dei gradi. Il log contiene l'andamento di gamma e le diagnostiche delle norme con prefisso `volumetric_`; il JSON conserva i valori finali/del best checkpoint.

Ogni job contiene inoltre `worker_command.json`, sufficiente a riprodurre la costruzione del checkpoint selezionato. Nessuna attention completa viene salvata durante il training.

## Export post-hoc delle attention VMA

Dopo la selezione:

```bash
python3 experiments/scripts/paper_comparison/txt_volumetric/export_attention.py \
  --run-dir RESULTS/seed_101/selected_test \
  --split test \
  --device cpu \
  --save-per-sample-npz
```

L'exporter ricostruisce train-only feature selection e grafo nell'ordine dei geni salvati, verifica l'allineamento, carica `best_model.pt` e abilita l'API di capture soltanto in evaluation. Produce:

- `attention_export/vma_attention_aggregate.csv`, aggregato per classe biologica, layer/encoder, head e arco diretto `target <- source`;
- opzionalmente `vma_attention_per_sample.npz`, con array sparse `[sample, layer, head, edge]` per pesi, volumi e logits;
- `attention_export_manifest.json`, con checkpoint, shape, beta, epsilon e diagnostiche.

L'NPZ conserva soltanto gli archi indotti HIPPIE; non contiene matrici dense gene-per-gene. Se checkpoint, geni, graph ordering o API di capture non sono disponibili/coerenti, l'exporter termina con un errore esplicito invece di produrre un file disallineato.

## Nota metodologica rispetto a GRAMformer

La formula volumetrica e il doppio gating seguono GRAMformer, ma il mapping usato qui è una estensione graph-aware specifica: per l'arco diretto `i <- j`, le colonne del volume sono `[query_i, key_i, key_j]`, cioè query, anchor e neighbor. Non è un'applicazione letterale delle modalità allineate originali di GRAMformer. Il termine TUPE della TxT, valutato sullo stesso arco, rende logits e pesi VMA dipendenti dall'espressione del paziente.

Riferimenti di progetto:

- `2606.06249v1.pdf` (GRAMformer), fornito separatamente;
- implementazione ufficiale: <https://github.com/ispamm/GRAMformer/blob/main/gramformer.py.py>;
- Koo et al., *Transcriptome Transformer* ([DOI](https://doi.org/10.1093/bib/bbaf628)) e relativo materiale supplementare.

Gli isolati non ricevono self-loop artificiali: il contributo VMA è nullo e attraversano esclusivamente la TxT densa. Gli score HIPPIE sono conservati nei manifest/export ma gli archi usati dal modello sono binari dopo la soglia.
