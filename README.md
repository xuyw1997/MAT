# MAT
> Code for paper ``Memory-Augmented Transformer for Efficient End-to-End Video Grounding''
## Data preparation

### ActivityNet Captions
Check this [web page](http://activity-net.org/download.html) and fill [this form](https://docs.google.com/forms/d/e/1FAIpQLSeKaFq9ZfcmZ7W0B0PbEhfbTHY41GeEgwsa7WobJgGUhn4DTQ/viewform) to get raw videos

### Charades-STA
download RGB frames from [here](https://prior.allenai.org/projects/charades)

### TACoS
Check this [web page](https://www.coli.uni-saarland.de/projects/smile/page.php?id=tacos) and fill in [this form](https://www.coli.uni-saarland.de/projects/smile/page.php?id=download) to get raw videos


### from raw videos to frames
run `extract_frames.sh`


## Train

run `python run_net.py --cfg=configs/xxx --train`
note that you need to replace the data paths in the configuration file with your own