#!/bin/sed -f
s/&amp;/\&/g
s/&lt;/</g
s/&gt;/>/g
s/&quot;/"/g
s/&apos;/'/g
s/&ldquo;/“/g
s/&rdquo;/”/g
s/１/1/g
s/２/2/g
s/３/3/g
s/４/4/g
s/５/5/g
s/６/6/g
s/７/7/g
s/８/8/g
s/９/9/g
s/０/0/g
s/ａ/a/g
s/ｂ/b/g
s/ｃ/c/g
s/ｄ/d/g
s/ｅ/e/g
s/ｆ/f/g
s/ｇ/g/g
s/ｈ/h/g
s/ｉ/i/g
s/ｊ/j/g
s/ｋ/k/g
s/ｌ/l/g
s/ｍ/m/g
s/ｎ/n/g
s/ｏ/o/g
s/ｐ/p/g
s/ｑ/q/g
s/ｒ/r/g
s/ｓ/s/g
s/ｔ/t/g
s/ｕ/u/g
s/ｖ/v/g
s/ｗ/w/g
s/ｘ/x/g
s/ｙ/y/g
s/ｚ/z/g
s/Ａ/A/g
s/Ｂ/B/g
s/Ｃ/C/g
s/Ｄ/D/g
s/Ｅ/E/g
s/Ｆ/F/g
s/Ｇ/G/g
s/Ｈ/H/g
s/Ｉ/I/g
s/Ｊ/J/g
s/Ｋ/K/g
s/Ｌ/L/g
s/Ｍ/M/g
s/Ｎ/N/g
s/Ｏ/O/g
s/Ｐ/P/g
s/Ｑ/Q/g
s/Ｒ/R/g
s/Ｓ/S/g
s/Ｔ/T/g
s/Ｕ/U/g
s/Ｖ/V/g
s/Ｗ/W/g
s/Ｘ/X/g
s/Ｙ/Y/g
s/Ｚ/Z/g
s/．/./g
s/·/./g
s/•/./g
s/·/./g
s/・/./g
s/％/%/g
s/＠/@/g
s/－/-/g
s/—/-/g
s/━/-/g
s/（/(/g
s/）/)/g
s/〈/(/g
s/〉/)/g
s/?\(?\+\)/<QMARKSPECIAL>\1/g
s/\(<QMARKSPECIAL>\)?/\1<QMARKSPECIAL>/g
s/?/？/g
s/<QMARKSPECIAL>/?/g
s/!/！/g
s/‘/'/g
s/’/'/g
s/『/'/g
s/』/'/g
s/「/“/g
s/」/”/g
s/《/“/g
s/》/”/g
s/"\([^"]*\)"/“\1”/g
s/■//g
s/＿//g
s/［/[/g
s/］/]/g
s/,\([^0-9]\| \|$\)/，\1/g
s/--[ \-]*/--/g
s/\*//g
s/[（(]金字旁容[）)]/镕/g
s///g
