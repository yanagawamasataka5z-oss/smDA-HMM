# Abe_minimal_metadata_template_ja_MY

Rendered from the workbook of the same name so it can be read on GitHub. **The `.xlsx` is the record**; this is a copy of its contents, verbatim — blanks and all. Regenerate with `scripts/ssbd_xlsx_to_md.py` after editing the workbook.

## Common

|  | Items | Class | Description | Description Notes |
|---|---|---|---|---|
| Project Information | Project name | Required | ssbd-repos-000536 | 管理者が付与します。カンリシャガフヨシマス |
|  | Project URL | Required | https://ssbd.riken.jp/repository/536/ | 管理者が付与します。カンリシャガフヨシマス |
|  | DOI | Recommended | https://doi.org/10.24631/ssbd.repos.2026.08.536 | 管理者が付与します。カンリシャガフヨシマス |
|  | Title | Required | Single-molecule imaging of 52 human RTKs on the surface of HEK293 cells | データセット全体を説明できる短文をご記述下さい。セツメイデキル タンブｎ キジュツクダサイ |
|  | Description | Required | raw movie data and results of single-particle tracking after hidden Markov model clustering | データセット全体を説明できる文章、キーワードなどをできるだけ広くご記述下さい。データセットに関わる論文の概要（abstract）でも問題ありません。デキルブンショウ ヒロク ゴキジュツクダサイ ロンブｎ ガイヨウ モンダイ |
|  | Release date | Required |  | データセットの公開日をご指定ください。 |
|  | License / Rights | Required | CC BY 4.0 | データセットに対するライセンスを、Creative Commons https://creativecommons.org/licenses/ を基準にご記述下さい。再利用を促進する観点から CC BY を、商用利用を禁止する場合は CC BY-NC-SA を推奨しております。キジュｎ ゴキジュツクダサイ サイリヨウヲ ソクシンスル カンテｎ ショウヨウリヨウ キンシスルバアイ スイショウシテオリマス |
|  | Funding information | Optional |  | データセットに関わる助成情報と助成番号をご記述ください。カカワル ジョセイジョウホウ ジョセイバンゴウ |
|  | Version | Optional | 1.0.0 | 管理者が付与します。データセットのバージョン、著者からのデータおよびメタデータの変更依頼の場合はmajor versionを変更します、管理側でのデータセットの追加・削除などの変更はminor versionを変更します、データセットの内容の変更を伴わない軽微な修正はrevisionを変更します。カンリシャガ フヨシマス データセットノバージョンチョシャカラノヘンコウイライ イライ ヘンコウ カンリガワ サクジョ ヘンコウ ヘンコウスル ナイヨウノヘンコウ トモナワナイ ケイビナシュウセイ ヘンコウシマス |
| Contact Information | Role | Required | Contact | 連絡先の方の役割（連絡先のみ、連絡先と画像データ作成の貢献者、連絡先と定量データ作成の貢献者、連絡先と画像データ作成および定量データ作成の貢献者）をお選びください。複数の連絡先を指定する場合には貢献者の欄に追記ください。レンラクサキ カタノ ヤクワリ レンラクサキノミ レンラクサキト ガゾウデータサクセイ サクセイノ コウケンシャ レンラクサキト テイリョウデータサクセイ コウケンシャ レンラクサキ ガゾウデータ サクセイ テイリョウデータサクセイ コウケンシャ フクスウノ レンラクサキヲ シテイスルバアイニハ コウケンシャ ラｎ ツイキクダサイ |
|  | First Name | Required | Yasushi | 連絡先のお名前（名前）をご記述下さい。ナマエ |
|  | Last Name | Required | Sako | 連絡先のお名前（苗字）をご記述下さい。レンラクサキノ オナマエ （ミョウジ |
|  | Contact E-mail | Required | sako@riken.jp | 連絡先のE-mailをご記述下さい、複数付けたい場合はコンマで区切って記述して下さい。 |
|  | Laboratory | Optional | Cellular Informatics Laboratory | 連絡先のご所属研究室をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニヒョウジサレマス レンラクサキノ ゴショゾク ケンキュウシツ ゴキジュツクダシア |
|  | Department | Optional |  | 連絡先のご所属部署をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニヒョウジサレマス レンラクサキゴ ブショヲ ゴキジュツクダサイ |
|  | Organization | Required | RIKEN | 連絡先のご所属組織をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニヒョウジサレマス レンラクサキノ ゴキジュツクダサイ |
|  | Address | Optional | 2-1, Hirosawa, Wako, 3510198, Japan | 連絡先のご住所をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニ ヒョウジサレマス レンラクサキノ ゴジュウショ ゴキジュツクダサイ |
|  | J-GLOBAL ID | Optional |  | 連絡先のJ-GLOBAL IDをご記述下さい。レンラクサキノ |
|  | researchmap ID | Optional |  | 連絡先のresearchmap IDをご記述下さいレンラクサキノ |
|  | ORCID | Optional | 0000-0002-5707-5455 | 連絡先のORCIDをご記述下さい。レンラクサキノ |
|  | Method Summary | Recommended | See details in Abe, et. al. (2026) bioRxiv. and GitHub, https://github.com/yanagawamasataka5z-oss/smDA-Igor, https://github.com/yanagawamasataka5z-oss/smDA-HMM | データセットの生成に利用された手法、キーワードなどをできるだけ広くご記述下さい。データセットに関わる論文のMethods、データセットに関わる論文の書誌情報でも問題ありません。例）See details in xxx, et. al. (2020) BioRxiv. セイセイ リヨウサレタ シュホウ ヒロク ゴキジュツクダサイ ショシジョウホウ モンダイ レイ |
|  | Paper DOI | Optional | 10.64898/2025.12.29.696957 | データセットに関わる論文にDOIが付与されている場合は、DOIを記述して下さい。（本データセットのDOIではありません）ロンブｎ フヨサレテイルバアイ キジュツシテクダサイ ホｎ |
|  | Paper URL | Recommended | https://www.biorxiv.org/content/10.64898/2025.12.29.696957v2 | データセットに関わる論文にURLが付与されている場合は、そのURLを記述して下さい。（本データセットのURLではありません）。SSBDプロジェクトでは生命動態情報と発生・細胞画像のデータを主に収集しているためPubMed ID/URL, BioRxiv等の記述を推奨しております。ロンブｎ フヨサレテイルバアイ キジュツシテクダサイ ホｎ セイメイドウタイジョウホウ ハッセイ サイボウ ガゾウ オモニ シュウシュウシテイルタメ トウキジュツヲ スイショウシテオリマス |
| Dataset Information | Data / File Formats | Required | tif movie files and csv result files | データセットに含まれるファイル形式です。ファイルの圧縮方式や、独自ファイルの内容などもご記述下さい。 |
|  | Data Size | Required | 995 GB | データセットの容量です。例）1.2 GB, 4 TBノヨウリョウデス レイ |
|  | Organism | Required | homo sapiens | データセットに含まれる生物種名を非省略の二名法（binomial name）でご記述下さい。不明な場合はUnknownとしてください。情報がない場合はNAとしてください。複数の生物の混合である場合はコンマで区切って記述して下さい。 例）Caenorhabditis elegans |
|  | Strain | Optional |  | データセットに含まれるStrainを、標準的な語彙で御記述下さい。不明な場合はUnknownとしてください。情報がない場合はNAとしてください。複数ある場合はコンマで区切って記述して下さい。例）BY4741 (MATa his3 delta 1 leu2 delta 0 met15 delta 0 ura3 delta 0) / Saccharomyces cerevisiae, w~{1118} (w~{1118};+;+;+) / Drosophila melanogasterニフクマレル ヒョウジュンテキナ ゴイ ゴキジュツクダサイ フクスウアルバアイハ クギッテキジュツシテクダサイ レイ |
|  | Cell Line | Optional | HEK293A | データセットに含まれるCell Lineを、標準的な語彙でご記述ください。不明な場合はUnknownとしてください。情報がない場合はNAとしてください。複数ある場合はコンマで区切って記述して下さい。例）HeLa cell, MDCK cell, MHH-ES-1 cell, CHO cellフクマレル ヒョウジュンテキナ キジュツ フクスウアルバアイ キジュツシテクダサイ レイ |
|  | Gene | Recommended |  | データセットで注目すべき遺伝子名（発現を操作している、変異が入っている、mRNA局在を観察しているなど）を標準的な表記でご記述ください。不明な場合はUnknownとしてください。情報が無い場合はNAとしてください。複数ある場合はコンマで区切って記述してください。例）par-2, par-3チュウモクスベキ イデンシメイ ハツゲン ソウサ ヘンイ ハイッテイル キョクザイ カンサツ ヒョウジュンテキナ ヒョウキデ フメイナバアイハ ジョウホウガナイバアイハ フクスウアルバアイハ レイ） |
|  | Protein | Recommended | human RTKs tagged with Halo protein at the c-term and labeld with SaraFluor 650 | データセットで注目すべきタンパク質名（観察している、発光させているなど）を標準的な表記でご記述ください。不明な場合はUnknownとしてください。情報が無い場合はNAとしてください。複数ある場合はコンマで区切って記述してください。例）Nanog, Oct4, H2Bデチュウモクスベキ メイ カンサツ ハッコウサセテイル ヒョウジュンテキ ヒョウキデ |
|  | Molecular Function (MF) | Optional | signal transduction | データセットに含まれるMolecular Functionを、標準的な語彙でご記述ください。Gene Ontology http://www.informatics.jax.org/vocab/gene_ontology/GO:0003674 が参考になります。不明な場合はUnknownとしてください。情報がない場合はNAとしてください。複数ある場合はコンマで区切って記述して下さい。例）translation activator activityフクマレル ヒョウジュンテキナゴイデ ヒョウジュンテキナ ヒョウゲンデ ゴキジュツクダサイ フクスウアルバアイハ キジュツシテクダサイ ゴイトシテ サンコウニナリマス レイ レイガホシイ |
|  | Biological Process (BP) | Optional |  | データセットに含まれるBiological Processを、標準的な語彙でご記述ください。Gene Ontology http://www.informatics.jax.org/vocab/gene_ontology/GO:0008150 が参考になります。不明な場合はUnknownとしてください。情報がない場合はNAとしてください。複数ある場合はコンマで区切って記述して下さい。例）cellular protein localization, egg activationフクマレルｙ ヒョウジュンテキナゴイ ゴキジュツクダサイ フクスウアルバアイハ ゴイトシテ サンコウニナリマス _x0000_<br>_x0003__x0007__x000E__x0002_ |
|  | Cellular Component (CC) | Optional | plasma membrane | データセットに含まれるCellular Componentを、標準的な語彙でご記述ください。Gene Ontology http://www.informatics.jax.org/vocab/gene_ontology/GO:0005575 が参考になります。不明な場合はUnknownとしてください。情報がない場合はNAとしてください。複数ある場合はコンマで区切って記述して下さい。例）cell, cytoplasm, kinetochore, chromosome, membraneフクマレル ヒョウジュンテキナゴイデ ヒョウジュンテキナヒョウゲンデ ゴキジュツクダサイ フクスウアルバアイハ キジュツシテクダサイ ゴイトシテ サンコウニナリマス レイ |
|  | Study Type | Optional |  | データセットに含まれる関わる生命科学用語を、標準的な語彙でご記述ください。MeSH https://www.ncbi.nlm.nih.gov/mesh が参考になります。複数ある場合はコンマで区切って記述して下さい。例）RNA Interference, Extracellular Signal-Regulated MAP Kinases, Fluorescence Resonance Energy Transfer , Single-Cell Analysis, Cell Differentiationカカワル セイメイカガクヨウゴ ヒョウジュンテキナ ゴイ ゴキジュツクダサイ サンコウニナリマス フクスウアルバアイ キジュツシテクダサイ レイ レイガホシイ |
|  | Imaging Methods | Optional | total internal reflection fluorescence microscopy | データセットに含まれる撮影方法を、標準的な語彙でご記述ください。Biological Imaging Methods Ontology https://bioportal.bioontology.org/ontologies/Fbbi が参考になります。複数ある場合はコンマで区切って記述して下さい。例）differential interference contrast microscopy, time lapse microscopy, scanning electron microscopy , transmission electron microscopy, spinning disk confocal microscopy, FRETフクマレル サツエイホウホウ ヒョウジュンテキナ ゴイデ サンコウニナリマス フクスウアルバアイハ キジュツシテクダサイ レイ レイガホシイ |
|  | Notes | Optional |  | データセットに関する補足事項などをご記述ください。ゴキジュツクダサイ |
|  | Templated version | Required | 2.12 | 管理者が付与します。データセットを入力するためのテンプレートのバージョンです。カンリシャガフヨシマス ニュウリョクスルタメノ |
| Contributor / Contact |  |  | Yasushi Sako | 画像データ作成・定量データ作成への貢献者のお名前をご記述下さい。ガゾウ・ テイリョウ コウケンシャ |
| Contact Information | Role | Required | Contact | 貢献者の役割（画像データ作成の貢献者、定量データ作成の貢献者、画像データ作成および定量データ作成の貢献者、など。）複数の連絡先を明記したい場合には、連絡先（Contact）の役割も加えてください。コウケンシャノ ヤクワリ ガゾウデータサクセイノコウケンシャ テイリョウデータ サクセイ コウケンシャ ガゾウデータサクセイ テイリョウデータサクセイ コウケンシャ フクスウノ レンラクサキヲ メイキシタイバアイニアｈ レンラクサキ ヤクワリ クワエテクダサイ |
|  | First Name | Required |  | 貢献者のお名前（名前）をご記述下さい。コウケンシャ ナマエ |
|  | Last Name | Required |  | 貢献者のお名前（苗字）をご記述下さい。レンラクサキノ オナマエ （ミョウジ |
|  | Contact E-mail | Optional |  | 貢献者のE-mailをご記述下さい、複数付けたい場合はコンマで区切って記述して下さい。 |
|  | Laboratory | Optional |  | 貢献者のご所属研究室をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニヒョウジサレマス レンラクサキノ ゴショゾク ケンキュウシツ ゴキジュツクダシア |
|  | Department | Optional |  | 貢献者のご所属部署をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニヒョウジサレマス レンラクサキゴ ブショヲ ゴキジュツクダサイ |
|  | Organization | Required |  | 貢献者のご所属組織をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニヒョウジサレマス レンラクサキノ ゴキジュツクダサイ |
|  | Address | Optional |  | 貢献者のご住所をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニ ヒョウジサレマス レンラクサキノ ゴジュウショ ゴキジュツクダサイ |
|  | J-GLOBAL ID | Optional |  | 貢献者のJ-GLOBAL IDをご記述下さい。レンラクサキノ |
|  | researchmap ID | Optional |  | 貢献者のresearchmap IDをご記述下さい |
|  | ORCID | Optional |  | 貢献者のORCIDをご記述下さい。レンラクサキノ |
| Contributor / Contact |  |  | Mitsuhiro Abe | 画像データ作成・定量データ作成への貢献者のお名前をご記述下さい。ガゾウ・ テイリョウ コウケンシャ |
| Contact Information | Role | Required | Contact | 貢献者の役割（画像データ作成の貢献者、定量データ作成の貢献者、画像データ作成および定量データ作成の貢献者、など。）複数の連絡先を明記したい場合には、連絡先（Contact）の役割も加えてください。コウケンシャノ ヤクワリ ガゾウデータサクセイノコウケンシャ テイリョウデータ サクセイ コウケンシャ ガゾウデータサクセイ テイリョウデータサクセイ コウケンシャ フクスウノ レンラクサキヲ メイキシタイバアイニアｈ レンラクサキ ヤクワリ クワエテクダサイ |
|  | First Name | Required | Mitsuhiro | 貢献者のお名前（名前）をご記述下さい。コウケンシャ ナマエ |
|  | Last Name | Required | Abe | 貢献者のお名前（苗字）をご記述下さい。レンラクサキノ オナマエ （ミョウジ |
|  | Contact E-mail | Optional | abemitsu@riken.jp | 貢献者のE-mailをご記述下さい、複数付けたい場合はコンマで区切って記述して下さい。 |
|  | Laboratory | Optional |  | 貢献者のご所属研究室をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニヒョウジサレマス レンラクサキノ ゴショゾク ケンキュウシツ ゴキジュツクダシア |
|  | Department | Optional |  | 貢献者のご所属部署をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニヒョウジサレマス レンラクサキゴ ブショヲ ゴキジュツクダサイ |
|  | Organization | Required | RIKEN | 貢献者のご所属組織をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニヒョウジサレマス レンラクサキノ ゴキジュツクダサイ |
|  | Address | Optional |  | 貢献者のご住所をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニ ヒョウジサレマス レンラクサキノ ゴジュウショ ゴキジュツクダサイ |
|  | J-GLOBAL ID | Optional |  | 貢献者のJ-GLOBAL IDをご記述下さい。レンラクサキノ |
|  | researchmap ID | Optional |  | 貢献者のresearchmap IDをご記述下さい |
|  | ORCID | Optional |  | 貢献者のORCIDをご記述下さい。レンラクサキノ |
| Contributor / Contact |  |  | Masataka Yanagawa | 画像データ作成・定量データ作成への貢献者のお名前をご記述下さい。ガゾウ・ テイリョウ コウケンシャ |
| Contact Information | Role | Required | Contact | 貢献者の役割（画像データ作成の貢献者、定量データ作成の貢献者、画像データ作成および定量データ作成の貢献者、など。）複数の連絡先を明記したい場合には、連絡先（Contact）の役割も加えてください。コウケンシャノ ヤクワリ ガゾウデータサクセイノコウケンシャ テイリョウデータ サクセイ コウケンシャ ガゾウデータサクセイ テイリョウデータサクセイ コウケンシャ フクスウノ レンラクサキヲ メイキシタイバアイニアｈ レンラクサキ ヤクワリ クワエテクダサイ |
|  | First Name | Required | Masataka | 貢献者のお名前（名前）をご記述下さい。コウケンシャ ナマエ |
|  | Last Name | Required | Yanagawa | 貢献者のお名前（苗字）をご記述下さい。レンラクサキノ オナマエ （ミョウジ |
|  | Contact E-mail | Optional | yanagawa@biophys.kyoto-u.ac.jp | 貢献者のE-mailをご記述下さい、複数付けたい場合はコンマで区切って記述して下さい。 |
|  | Laboratory | Optional |  | 貢献者のご所属研究室をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニヒョウジサレマス レンラクサキノ ゴショゾク ケンキュウシツ ゴキジュツクダシア |
|  | Department | Optional |  | 貢献者のご所属部署をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニヒョウジサレマス レンラクサキゴ ブショヲ ゴキジュツクダサイ |
|  | Organization | Required | Kyoto University | 貢献者のご所属組織をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニヒョウジサレマス レンラクサキノ ゴキジュツクダサイ |
|  | Address | Optional |  | 貢献者のご住所をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニ ヒョウジサレマス レンラクサキノ ゴジュウショ ゴキジュツクダサイ |
|  | J-GLOBAL ID | Optional |  | 貢献者のJ-GLOBAL IDをご記述下さい。レンラクサキノ |
|  | researchmap ID | Optional |  | 貢献者のresearchmap IDをご記述下さい |
|  | ORCID | Optional |  | 貢献者のORCIDをご記述下さい。レンラクサキノ |
| Contributor / Contact |  |  |  | 画像データ作成・定量データ作成への貢献者のお名前をご記述下さい。ガゾウ・ テイリョウ コウケンシャ |
| Contact Information | Role | Required | Quantitative dataset contributor | 貢献者の役割（画像データ作成の貢献者、定量データ作成の貢献者、画像データ作成および定量データ作成の貢献者、など。）複数の連絡先を明記したい場合には、連絡先（Contact）の役割も加えてください。コウケンシャノ ヤクワリ ガゾウデータサクセイノコウケンシャ テイリョウデータ サクセイ コウケンシャ ガゾウデータサクセイ テイリョウデータサクセイ コウケンシャ フクスウノ レンラクサキヲ メイキシタイバアイニアｈ レンラクサキ ヤクワリ クワエテクダサイ |
|  | First Name | Required |  | 貢献者のお名前（名前）をご記述下さい。コウケンシャ ナマエ |
|  | Last Name | Required |  | 貢献者のお名前（苗字）をご記述下さい。レンラクサキノ オナマエ （ミョウジ |
|  | Contact E-mail | Optional |  | 貢献者のE-mailをご記述下さい、複数付けたい場合はコンマで区切って記述して下さい。 |
|  | Laboratory | Optional |  | 貢献者のご所属研究室をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニヒョウジサレマス レンラクサキノ ゴショゾク ケンキュウシツ ゴキジュツクダシア |
|  | Department | Optional |  | 貢献者のご所属部署をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニヒョウジサレマス レンラクサキゴ ブショヲ ゴキジュツクダサイ |
|  | Organization | Required |  | 貢献者のご所属組織をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニヒョウジサレマス レンラクサキノ ゴキジュツクダサイ |
|  | Address | Optional |  | 貢献者のご住所をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニ ヒョウジサレマス レンラクサキノ ゴジュウショ ゴキジュツクダサイ |
|  | J-GLOBAL ID | Optional |  | 貢献者のJ-GLOBAL IDをご記述下さい。レンラクサキノ |
|  | researchmap ID | Optional |  | 貢献者のresearchmap IDをご記述下さい |
|  | ORCID | Optional |  | 貢献者のORCIDをご記述下さい。レンラクサキノ |
| Contributor / Contact |  |  |  | 画像データ作成・定量データ作成への貢献者のお名前をご記述下さい。ガゾウ・ テイリョウ コウケンシャ |
| Contact Information | Role | Required | Quantitative dataset contributor | 貢献者の役割（画像データ作成の貢献者、定量データ作成の貢献者、画像データ作成および定量データ作成の貢献者、など。）複数の連絡先を明記したい場合には、連絡先（Contact）の役割も加えてください。コウケンシャノ ヤクワリ ガゾウデータサクセイノコウケンシャ テイリョウデータ サクセイ コウケンシャ ガゾウデータサクセイ テイリョウデータサクセイ コウケンシャ フクスウノ レンラクサキヲ メイキシタイバアイニアｈ レンラクサキ ヤクワリ クワエテクダサイ |
|  | First Name | Required |  | 貢献者のお名前（名前）をご記述下さい。コウケンシャ ナマエ |
|  | Last Name | Required |  | 貢献者のお名前（苗字）をご記述下さい。レンラクサキノ オナマエ （ミョウジ |
|  | Contact E-mail | Optional |  | 貢献者のE-mailをご記述下さい、複数付けたい場合はコンマで区切って記述して下さい。 |
|  | Laboratory | Optional |  | 貢献者のご所属研究室をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニヒョウジサレマス レンラクサキノ ゴショゾク ケンキュウシツ ゴキジュツクダシア |
|  | Department | Optional |  | 貢献者のご所属部署をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニヒョウジサレマス レンラクサキゴ ブショヲ ゴキジュツクダサイ |
|  | Organization | Required |  | 貢献者のご所属組織をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニヒョウジサレマス レンラクサキノ ゴキジュツクダサイ |
|  | Address | Optional |  | 貢献者のご住所をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニ ヒョウジサレマス レンラクサキノ ゴジュウショ ゴキジュツクダサイ |
|  | J-GLOBAL ID | Optional |  | 貢献者のJ-GLOBAL IDをご記述下さい。レンラクサキノ |
|  | researchmap ID | Optional |  | 貢献者のresearchmap IDをご記述下さい |
|  | ORCID | Optional |  | 貢献者のORCIDをご記述下さい。レンラクサキノ |
| Contributor / Contact |  |  |  | 画像データ作成・定量データ作成への貢献者のお名前をご記述下さい。ガゾウ・ テイリョウ コウケンシャ |
| Contact Information | Role | Required | Quantitative dataset contributor | 貢献者の役割（画像データ作成の貢献者、定量データ作成の貢献者、画像データ作成および定量データ作成の貢献者、など。）複数の連絡先を明記したい場合には、連絡先（Contact）の役割も加えてください。コウケンシャノ ヤクワリ ガゾウデータサクセイノコウケンシャ テイリョウデータ サクセイ コウケンシャ ガゾウデータサクセイ テイリョウデータサクセイ コウケンシャ フクスウノ レンラクサキヲ メイキシタイバアイニアｈ レンラクサキ ヤクワリ クワエテクダサイ |
|  | First Name | Required |  | 貢献者のお名前（名前）をご記述下さい。コウケンシャ ナマエ |
|  | Last Name | Required |  | 貢献者のお名前（苗字）をご記述下さい。レンラクサキノ オナマエ （ミョウジ |
|  | Contact E-mail | Optional |  | 貢献者のE-mailをご記述下さい、複数付けたい場合はコンマで区切って記述して下さい。 |
|  | Laboratory | Optional |  | 貢献者のご所属研究室をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニヒョウジサレマス レンラクサキノ ゴショゾク ケンキュウシツ ゴキジュツクダシア |
|  | Department | Optional |  | 貢献者のご所属部署をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニヒョウジサレマス レンラクサキゴ ブショヲ ゴキジュツクダサイ |
|  | Organization | Required |  | 貢献者のご所属組織をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニヒョウジサレマス レンラクサキノ ゴキジュツクダサイ |
|  | Address | Optional |  | 貢献者のご住所をご記述下さい。複数付けたい場合はコンマで区切って記述して下さい。ジドウテキニ ヒョウジサレマス レンラクサキノ ゴジュウショ ゴキジュツクダサイ |
|  | J-GLOBAL ID | Optional |  | 貢献者のJ-GLOBAL IDをご記述下さい。レンラクサキノ |
|  | researchmap ID | Optional |  | 貢献者のresearchmap IDをご記述下さい |
|  | ORCID | Optional |  | 貢献者のORCIDをご記述下さい。レンラクサキノ |

## Dropdowns

| Licenses | Contact Role | Contributors Role |
|---|---|---|
| CC BY 4.0 | Contact | Contact |
| CC BY-NC-ND 4.0 | Contact,Image dataset contributor | Contact,Image dataset contributor |
| CC BY-NC-SA 4.0 | Contact,Quantitative data contributor | Contact,Quantitative data contributor |
| CC BY-NC 4.0 | Contact,Image dataset contributor,Quantitative dataset contributor | Contact,Image dataset contributor,Quantitative dataset contributor |
| CC0 1.0 Universal |  | Image dataset contributor |
| CC BY-SA 4.0 |  | Quantitative dataset contributor |
| CC BY-ND 4.0 |  | Image dataset contributor,Quantitative data contributor |
