# IA Teste — Detecção de Intrusão em Rede com Machine Learning

Sistema de detecção de intrusão (IDS) em duas camadas, treinado em tráfego de rede real e servido por uma API.

- **Camada A — supervisionada:** classificador multiclasse (LightGBM) que identifica *qual* ataque está acontecendo.
- **Camada B — não supervisionada:** autoencoder treinado apenas em tráfego benigno, que sinaliza o que nunca foi visto antes.

> Status: em desenvolvimento.

---

## Por que duas camadas

Um classificador supervisionado só reconhece ataques que estavam no dataset. Um detector de anomalia não sabe nomear o ataque, mas percebe desvios do normal. Juntos cobrem os dois casos: ameaça conhecida e ameaça nova.

## A métrica que importa

Acurácia não é reportada aqui de propósito. Em detecção de intrusão o tráfego benigno é a maioria esmagadora, então prever "tudo normal" já dá mais de 99% de acurácia.

O que este projeto otimiza é a **taxa de falso positivo (FPR)**. Um SOC que processa 1 milhão de fluxos por dia com um modelo de 1% de FPR recebe 10.000 alertas falsos diários — e o sistema vira ruído.

| Métrica | Alvo | Resultado |
|---|---|---|
| F1 macro | > 0,90 | _a preencher_ |
| Recall por classe | > 0,85 | _a preencher_ |
| FPR | < 0,1 % | _a preencher_ |
| PR-AUC | > 0,95 | _a preencher_ |
| Latência p95 | < 20 ms | _a preencher_ |

## Dataset

CIC-IDS2017 — **em versão corrigida**. O dataset original tem erros de rotulagem e bugs de extração de features documentados na literatura; versões corrigidas (LYCOS-IDS2017 e a *improved* do WTMC2021) resolvem parte deles.

Tratamentos aplicados no `prepare_data.py`:

- remoção de linhas duplicadas
- tratamento de `Inf` / `NaN` em `Flow Bytes/s` e `Flow Packets/s`
- remoção de features com vazamento (IPs, timestamps, IDs de fluxo)
- **split temporal**, não aleatório — fluxos de um mesmo ataque são quase idênticos, e um split aleatório coloca cópias do treino no teste

## Validação em tráfego próprio

Além do dataset público, o modelo é testado em tráfego gerado em laboratório:

1. VM Kali (atacante) e VM Ubuntu com HTTP + SSH (alvo), em rede interna
2. tráfego benigno: navegação, download, SSH legítimo
3. ataques: `nmap -sS` (port scan), `hydra` (brute force SSH), `slowloris` (DoS)
4. captura com `tcpdump`, extração de fluxos com CICFlowMeter
5. inferência **sem retreinar**

Resultados desta etapa: _a preencher_.

## Estrutura

```
src/prepare_data.py    limpeza, split temporal, escrita em Parquet
src/train_supervised.py  LightGBM multiclasse + SHAP
src/train_autoencoder.py autoencoder + calibração de limiar
api/main.py            FastAPI: /predict e /stream
```

## Como rodar

```bash
pip install -r requirements.txt
python src/prepare_data.py --input data/raw --output data/processed
python src/train_supervised.py
python src/train_autoencoder.py
uvicorn api.main:app --reload
```

## Stack

Python · pandas · scikit-learn · LightGBM · PyTorch · SHAP · FastAPI · Docker

## Limitações

Este é um projeto de estudo. O modelo foi validado em dataset público e em laboratório controlado — não foi testado em rede de produção, onde a distribuição do tráfego é diferente e a degradação seria significativa.

## Referências

- [CIC-IDS2017 — Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
- [Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study (WTMC 2021)](https://intrusion-detection.distrinet-research.be/WTMC2021/extended_doc.html)
- [From CIC-IDS2017 to LYCOS-IDS2017: A corrected dataset](https://dl.acm.org/doi/10.1145/3486622.3493973)
- [Network intrusion datasets: a survey, limitations, and recommendations (2025)](https://dl.acm.org/doi/10.1016/j.cose.2025.104510)

---

Robert Kevyn Gomes — Ciência da Computação, Instituto Mauá de Tecnologia
