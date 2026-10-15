# Editable figure sources and replay

The final manuscript uses the English exports archived in `figures/v3.5.0/`. All twelve manuscript PDFs were compared with the English companion; eleven are permitted here. Figure 2 is intentionally excluded because it contains upstream source excerpts.

Figure 1 has its author-generated native `figure1.drawio` source alongside PDF, SVG and PNG exports. Open the native file in a compatible diagrams.net/draw.io editor to edit the diagram. No editor implementation or third-party icon library is bundled. This diagram illustrates fixed rankings, target mapping, evaluability and shared-sample score comparison; its positions and small observation pairs are schematic.

Figure S1's `figureS1.svg` is its editable-text vector source. The English labels retain the existing sample counts and graph relationships. A compatible SVG editor can edit labels; validate fonts and clipping after re-export. The former internal translation script depended on private manuscript paths and is not redistributed.

`build_public_figures.py` copies these two reviewed schematic export sets and regenerates the other nine statistical figures from frozen public JSON. It produces 33 figure exports and matching numerical tables without new training, bootstrap estimation, sample selection, or significance testing. Copying a schematic is disclosed rather than presented as a new analysis. Re-exporting the editable sources can change fonts or metadata and need not produce byte-identical files.
