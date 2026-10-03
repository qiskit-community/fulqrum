# Vendored third-party headers

These headers are copied from other projects rather than consumed as
dependencies. Each is compiled into Fulqrum's extension modules and installed
alongside them, so its license text is kept in this directory.

| File | Upstream | Version | License |
| --- | --- | --- | --- |
| `json.hpp` | [nlohmann/json](https://github.com/nlohmann/json) | 3.12.0 | MIT (`LICENSE-nlohmann-json.txt`) |
| `hash_table8.hpp`, `hash_set8.hpp` | [ktprime/emhash](https://github.com/ktprime/emhash) | 1.7.2, 1.7.4 | MIT (`LICENSE-emhash.txt`) |
| `rapidhash.h` | [Nicoshev/rapidhash](https://github.com/Nicoshev/rapidhash) | — | MIT (`LICENSE-rapidhash.txt`) |
| `pstream.h` | [PStreams](https://pstreams.sourceforge.net/) | — | BSL-1.0 (`LICENSE-pstreams.txt`) |

`hash_table8.hpp` and `hash_set8.hpp` carry local modifications. They are not
byte-identical to any upstream release, so the versions above are the ones
declared in each file's own header comment.

A subset of [Boost](https://www.boost.org/) 1.89 headers is vendored separately
under `fulqrum/include/boost`, with its license in
`fulqrum/include/boost/LICENSE_1_0.txt`.

`qiskit-addon-sqd-hpc` is a git submodule rather than a vendored copy, so its
headers and its `LICENSE.txt` both come from that repository; `MANIFEST.in`
includes the license so it reaches the sdist alongside the headers.
