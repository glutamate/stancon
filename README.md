# Paper for submission to StanCon

## Build instructions

1. clone this repository (stancon) and the open repository (https://github.com/diffusionkinetics/open) in the same directory (see how in stack.yaml the subdirectories in the open repository are referenced).
2. Install Stan (see below)
3. `cd stancon`
4. `stack install`
5. `stancon >stancon.html`
6. now you can open `stancon.html` in a web browser. For instance, you could enter at the command line: `firefox stancon.html’

The paper is using the inliterate system (https://github.com/diffusionkinetics/open/tree/master/inliterate)

## Installing Stan

you will need various compiler infrastructure, for instance clang and libc++-dev.

```
wget https://github.com/stan-dev/cmdstan/releases/download/v2.17.0/cmdstan-2.17.0.tar.gz
tar -xzvf cmdstan-2.17.0.tar.gz
sudo mv cmdstan-2.17.0 /opt/stan
(cd /opt/stan && sudo make build -j4)
rm cmdstan-2.17.0.tar.gz
```