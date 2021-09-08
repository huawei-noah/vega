# model zoo

## Image Classification on Cifar10

| Model | Model Size(M) | Flops(M) | Top-1 ACC | Inference Time(μs) | Inference Device | Download |
|---|---|---|---|---|---|:--:|
| CARS-A | 7.72 | 469 | 95.923477 | 51.28 | V100 | [tar](https://box.saas.huaweicloud.com/p/bef2bcf575db82eeaf0b344b04a27002) |
| CARS-B | 8.45 | 548 | 96.584535 | 69.00 | V100 | [tar](https://box.saas.huaweicloud.com/p/868026b6e897fc5d95cc4afafd31e5a2) |
| CARS-C | 9.32 | 620 | 96.744791 | 71.62 | V100 | [tar](https://box.saas.huaweicloud.com/p/683a63d576a52f1229aecdcd646b0ea9) |
| CARS-D | 10.5 | 729 | 97.055288 | 82.72 | V100 | [tar](https://box.saas.huaweicloud.com/p/36c558492319c3b362110bedc96d9680) |
| CARS-E | 11.3 | 786 | 97.245592 | 88.98 | V100 | [tar](https://box.saas.huaweicloud.com/p/1f5c292d15bb29d2b82664691d07b9cb) |
| CARS-F | 16.7 | 1234 | 97.295673 | 244.07 | V100 | [tar](https://box.saas.huaweicloud.com/p/846a7460c15699cb95d6f43f5e9d0598) |
| CARS-G | 19.1 | 1439 | 97.375801 | 391.20 | V100 | [tar](https://box.saas.huaweicloud.com/p/1d8bec17e7a4a58474631bdca7c588ce) |
| CARS-H | 19.7 | 1456 | 97.425881 | 398.88 | V100 | [tar](https://box.saas.huaweicloud.com/p/5b0e5dc2fc9160559ed214d5caa0402b) |

## Image Classification on ImageNet

| Model | Model Size(M) | Flops(G) | Top-1 ACC | Inference Time(s) | Download |
|---|---|---|---|---|:--:|
| EfficientNet:B0 | 20.3 | 0.40 | 76.82 | 0.0088s/iters | [tar](https://box.saas.huaweicloud.com/p/24c9e4ae3a74c361a5b34ed3b37def29) |
| EfficientNet:B4 | 74.3 | 4.51 | 82.87 | 0.015s/iters | [tar](https://box.saas.huaweicloud.com/p/256d3b2dffdcf9bcf317dbe5554ed604) |
| EfficientNet:B8:672 | 88 | 63 | 85.7 | 0.98s/iters | [tar](https://box.saas.huaweicloud.com/p/f49bc1afd817562deb1e6d461d8cebfd) |
| EfficientNet:B8:800 | 88 | 97 | 85.8 | 1.49s/iters | [tar](https://box.saas.huaweicloud.com/p/287cb98d96152053d24fb7afc3e9e0be) |

## Detection on COCO-minival

| Model | Model Size(M) | Flops(G) | mAP | Inference Time(ms) | Inference Device | Download |
|---|---|---|---|---|---|:--:|
| SM-NAS:E0 | 16.23 | 22.01 | 27.11 | 24.56 | V100 | [tar](https://box.saas.huaweicloud.com/p/2db2953a19106ca0de3bad4d96cefd1d) |
| SM-NAS:E1 | 37.83 | 64.72 | 34.20 | 32.07 | V100 | [tar](https://box.saas.huaweicloud.com/p/3e0691a8e820cad41ad4ac01bc7d13b3) |
| SM-NAS:E2 | 33.02 | 77.04 | 40.04 | 39.50 | V100 | [tar](https://box.saas.huaweicloud.com/p/c4212a0daab666298bb46373bfdde9e4) |
| SM-NAS:E3 | 52.05 | 116.22 | 42.68 | 50.71 | V100 | [tar](https://box.saas.huaweicloud.com/p/840164137ef02ea5d5072c8e577ac663) |
| SM-NAS:E4 | 92 | 115.51 | 43.89 | 80.22 | V100 | [tar](https://box.saas.huaweicloud.com/p/85ed655da827564752ed460119416ef3) |
| SM-NAS:E5 | 90.47 | 249.14 | 46.05 | 108.07 | V100 | [tar](https://box.saas.huaweicloud.com/p/252a999404f1f93d4c953ac1b13aebad) |

## Detection on CULane

| Model | Flops(G) | F1 Score | Inference Time(ms) | Inference Device | Download |
|---|---|---|---|---|:--:|
| AutoLane: CULane-s | 66.5 | 71.5 | - | V100 | [tar](https://box.saas.huaweicloud.com/p/18bcca1d2e3f19bcf52e1408a0853931) |
| AutoLane: CULane-m | 66.9 | 74.6 | - | V100 | [tar](https://box.saas.huaweicloud.com/p/398d76084a3a89c656cc2671a9edab12) |
| AutoLane: CULane-l | 273 | 75.2 | - | V100 | [tar](https://box.saas.huaweicloud.com/p/e4bd2cba73fee6012bbc4ef9ed3699b9) |

## Super-Resolution on Urban100, B100, Set14, Set5

<table>
    <tr align="center">
        <td rowspan="2"><b>Model</td>
        <td rowspan="2"><b>Model Size(M)</td>
        <td rowspan="2"><b>Flops(G)</td>
        <td colspan="2"><b>Urban100</td>
        <td colspan="2"><b>B100</td>
        <td colspan="2"><b>Set14</td>
        <td colspan="2"><b>Set5</td>
        <td rowspan="2"><b>Inference Time(ms)</td>
        <td rowspan="2"><b>Download</td>
    </tr>
    <tr>
        <td><b>PSNR</td>
        <td><b>SSIM</td>
        <td><b>PSNR</td>
        <td><b>SSIM</td>
        <td><b>PSNR</td>
        <td><b>SSIM</td>
        <td><b>PSNR</td>
        <td><b>SSIM</td>
    </tr>
    <tr>
        <td>ESR-EA:ESRN-V-1</td>
        <td>1.32</td>
        <td>40.616</td>
        <td>31.65</td>
        <td>0.8814</td>
        <td>32.09</td>
        <td>0.8802</td>
        <td>33.37</td>
        <td>0.8887</td>
        <td>37.79</td>
        <td>0.9566</td>
        <td>29.38</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/c425b83d509c42a11a0d3687e04a3a61">tar</a></td>
    </tr>
    <tr>
        <td>ESR-EA:ESRN-V-2</td>
        <td>1.31</td>
        <td>40.21</td>
        <td>31.69</td>
        <td>0.8829</td>
        <td>32.08</td>
        <td>0.8810</td>
        <td>33.37</td>
        <td>0.8911</td>
        <td>37.84</td>
        <td>0.9569</td>
        <td>31.25</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/d95700836ad45ab7e8278a370179ba73">tar</a></td>
    </tr>
        <tr>
        <td>ESR-EA:ESRN-V-3</td>
        <td>1.31</td>
        <td>41.676</td>
        <td>31.47</td>
        <td>0.8803</td>
        <td>32.05</td>
        <td>0.8789</td>
        <td>33.35</td>
        <td>0.8878</td>
        <td>37.79</td>
        <td>0.9570</td>
        <td>21.78</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/9ad38ae4122c47e421fc0877031d65ea">tar</a></td>
    </tr>
        <tr>
        <td>ESR-EA:ESRN-V-4</td>
        <td>1.35</td>
        <td>40.17</td>
        <td>31.58</td>
        <td>0.8814</td>
        <td>32.06</td>
        <td>0.8810</td>
        <td>33.35</td>
        <td>0.0.8902</td>
        <td>37.83</td>
        <td>0.9567</td>
        <td>30.98</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/3a107e4f0c79f8f8b78363978846972a">tar</a></td>
    </tr>
    <tr>
        <td>SR_EA:M2Mx2-A</td>
        <td>3.20</td>
        <td>196.27</td>
        <td>32.20</td>
        <td>0.8948</td>
        <td>32.20</td>
        <td>0.8842</td>
        <td>33.65</td>
        <td>0.8943</td>
        <td>38.06</td>
        <td>0.9588</td>
        <td>11.41</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/3e5ed5c2453ac0a3d21794ab30bdbe04">tar</a></td>
    </tr>
    <tr>
        <td>SR_EA:M2Mx2-B</td>
        <td>0.61</td>
        <td>35.03</td>
        <td>31.77</td>
        <td>0.8796</td>
        <td>32.00</td>
        <td>0.8989</td>
        <td>33.32</td>
        <td>0.8870</td>
        <td>37.73</td>
        <td>0.9562</td>
        <td>8.55</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/e1bee445b3e06c373304fb0e0f56907e">tar</a></td>
    </tr>
    <tr>
        <td>SR_EA:M2Mx2-C</td>
        <td>0.24</td>
        <td>13.49</td>
        <td>30.92</td>
        <td>0.8717</td>
        <td>31.89</td>
        <td>0.8783</td>
        <td>33.13</td>
        <td>0.8829</td>
        <td>37.56</td>
        <td>0.9556</td>
        <td>5.59</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/741abb8eab5d47eb8933a0ebb46a27c5">tar</a></td>
    </tr>
<table>

## Segmentation on VOC2012

| Model | Model Size(M) | Flops(G) | Params(K) | mIOU | Download |
|---|---|---|---|---|:--:|
| Adelaide | 10.6 | 0.5784 | 3822.14 | 0.7602 |[tar](https://box.saas.huaweicloud.com/p/4d4eadea8c1d17af2e224ba094d70c0c) |

## DNet

### Part 1

<table>
    <caption align="center">
        <th rowspan="2"><b>Model</th>
        <th rowspan="2"><b>Accuracy</th>
        <th rowspan="2"><b>FLOPS (G)</th>
        <th rowspan="2"><b>Params (M)</th>
        <th colspan="6"><b>Inference Time (ms)</th>
        <td rowspan="2"><b>Download</td>
    </caption>
    <tr>
        <th>D-224p</th>
        <th>D-512p</th>
        <th>D-720p</th>
        <th>Pytorch-V100</th>
        <th>Caffe-V100</th>
        <th>Caffe-CPU</th>
    </tr>
    <tr>
        <td>D-Net-21</td>
        <td>61.51</td>
        <td>0.21</td>
        <td>2.59</td>
        <td>2.02</td>
        <td>2.84</td>
        <td>6.11</td>
        <td>3.02</td>
        <td>2.26</td>
        <td>22.30</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/a954358ecf9f6f6e7f691f4fd7fe0521">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-24</td>
        <td>62.06</td>
        <td>0.24</td>
        <td>2.62</td>
        <td>1.98</td>
        <td>2.77</td>
        <td>6.60</td>
        <td>2.89</td>
        <td>2.51</td>
        <td>23.50</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/a7e57543abcbb091f62d2bbf9c3a6b6b">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-30</td>
        <td>64.49</td>
        <td>0.30</td>
        <td>3.16</td>
        <td>1.95</td>
        <td>2.81</td>
        <td>7.22</td>
        <td>2.86</td>
        <td>2.71</td>
        <td>27.30</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/b6f1cfcc5606e2fe1cdecdcabcce0d04">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-39</td>
        <td>67.71</td>
        <td>0.39</td>
        <td>4.37</td>
        <td>2.06</td>
        <td>3.01</td>
        <td>7.12</td>
        <td>3.10</td>
        <td>2.70</td>
        <td>23.80</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/4762b7d7d8a241dbc22402e243372e3d">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-40</td>
        <td>66.92</td>
        <td>0.40</td>
        <td>3.94</td>
        <td>1.98</td>
        <td>2.96</td>
        <td>6.42</td>
        <td>2.97</td>
        <td>2.48</td>
        <td>17.60</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/14a56ca2c26a9364078d33fc50f15364">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-59</td>
        <td>71.29</td>
        <td>0.59</td>
        <td>7.13</td>
        <td>2.30</td>
        <td>3.31</td>
        <td>8.10</td>
        <td>3.28</td>
        <td>2.71</td>
        <td>33.10</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/909410975f67c0c8e44aaeb7d96b2b51">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-94</td>
        <td>70.23</td>
        <td>0.94</td>
        <td>5.80</td>
        <td>2.19</td>
        <td>3.54</td>
        <td>8.75</td>
        <td>2.93</td>
        <td>2.84</td>
        <td>39.10</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/0909e8bbf64a9e6c544b87b4bd038daa">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-124</td>
        <td>76.09</td>
        <td>1.24</td>
        <td>11.80</td>
        <td>2.87</td>
        <td>5.09</td>
        <td>15.42</td>
        <td>4.36</td>
        <td>3.65</td>
        <td>56.30</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/fb0c6df19486589e6fa97aa5138d22b1">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-147</td>
        <td>71.91</td>
        <td>1.47</td>
        <td>10.47</td>
        <td>2.50</td>
        <td>5.28</td>
        <td>23.84</td>
        <td>2.29</td>
        <td>2.24</td>
        <td>45.20</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/02ad969112a5c3637d4483b9033ee02d">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-156</a></td>
        <td>74.46</td>
        <td>1.56</td>
        <td>15.24</td>
        <td>2.52</td>
        <td>4.13</td>
        <td>11.89</td>
        <td>3.02</td>
        <td>2.86</td>
        <td>32.40</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/2d2ad0b637a9025de1e31b7e8b4f44c0">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-159</a></td>
        <td>75.21</td>
        <td>1.59</td>
        <td>19.13</td>
        <td>3.05</td>
        <td>4.71</td>
        <td>12.76</td>
        <td>4.55</td>
        <td>3.78</td>
        <td>43.30</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/f270ebf0b4b08e05e7c68604329d832c">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-166</a></td>
        <td>72.93</td>
        <td>1.66</td>
        <td>10.82</td>
        <td>2.27</td>
        <td>4.26</td>
        <td>11.42</td>
        <td>2.97</td>
        <td>2.68</td>
        <td>50.60</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/af25c6ed235e65dccb350568bb26564d">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-167</a></td>
        <td>74.18</td>
        <td>1.67</td>
        <td>10.56</td>
        <td>2.51</td>
        <td>4.21</td>
        <td>12.03</td>
        <td>2.92</td>
        <td>2.84</td>
        <td>43.60</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/4b3f36b98ab29f6dbab63160e0687ae2">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-172</a></td>
        <td>76.41</td>
        <td>1.72</td>
        <td>17.44</td>
        <td>4.02</td>
        <td>10.72</td>
        <td>36.41</td>
        <td>3.51</td>
        <td>34.33</td>
        <td>106.20</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/af89baf4fabc6cfdd2d544f9c0e4f1cd">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-177</a></td>
        <td>75.55</td>
        <td>1.77</td>
        <td>19.48</td>
        <td>3.65</td>
        <td>5.40</td>
        <td>14.09</td>
        <td>5.66</td>
        <td>4.64</td>
        <td>58.80</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/8440153902249e3abfda436057b02950">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-234</a></td>
        <td>78.80</td>
        <td>2.34</td>
        <td>28.45</td>
        <td>5.03</td>
        <td>8.01</td>
        <td>21.35</td>
        <td>8.69</td>
        <td>7.44</td>
        <td>87.10</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/a8159dfe99dd52bae5703e7de0f6639f">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-263</a></td>
        <td>76.87</td>
        <td>2.63</td>
        <td>21.42</td>
        <td>3.42</td>
        <td>6.04</td>
        <td>19.13</td>
        <td>4.44</td>
        <td>4.08</td>
        <td>90.40</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/26ff769019596bd2e180381186261a3d">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-264</a></td>
        <td>76.52</td>
        <td>2.64</td>
        <td>20.17</td>
        <td>3.12</td>
        <td>5.54</td>
        <td>16.88</td>
        <td>4.27</td>
        <td>4.01</td>
        <td>62.50</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/327466cea4e33b03d921005056d4c213">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-275</a></td>
        <td>78.28</td>
        <td>2.75</td>
        <td>30.76</td>
        <td>4.09</td>
        <td>10.56</td>
        <td>34.76</td>
        <td>4.22</td>
        <td>4.03</td>
        <td>96.60</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/dcd4216da8acb5cf69337c914d44875a">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-367</a></td>
        <td>79.37</td>
        <td>3.67</td>
        <td>41.83</td>
        <td>5.56</td>
        <td>15.09</td>
        <td>66.57</td>
        <td>6.86</td>
        <td>6.05</td>
        <td>130.90</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/df70f5ddbd5ebcd8634a51d4a19c1cb0">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-394</a></td>
        <td>77.91</td>
        <td>3.94</td>
        <td>25.15</td>
        <td>3.35</td>
        <td>7.79</td>
        <td>24.97</td>
        <td>4.38</td>
        <td>4.12</td>
        <td>75.80</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/5b75df5d8c7192f5a1089768a98ab304">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-426</a></td>
        <td>78.24</td>
        <td>4.24</td>
        <td>25.31</td>
        <td>3.56</td>
        <td>8.37</td>
        <td>27.06</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/8f0e0f9be21cc93f8123f9a70ada18f8">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-504</a></td>
        <td>78.96</td>
        <td>5.04</td>
        <td>28.47</td>
        <td>3.57</td>
        <td>9.07</td>
        <td>30.32</td>
        <td>4.59</td>
        <td>4.90</td>
        <td>93.50</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/01f3337b46900322dc99a8617ebfdbbb">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-538</a></td>
        <td>80.92</td>
        <td>5.38</td>
        <td>44.00</td>
        <td>5.90</td>
        <td>13.89</td>
        <td>46.83</td>
        <td>9.73</td>
        <td>8.52</td>
        <td>156.80</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/3572182d1011fffd26aab985de9a505b">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-572</a></td>
        <td>80.41</td>
        <td>5.72</td>
        <td>49.29</td>
        <td>6.24</td>
        <td>18.56</td>
        <td>87.48</td>
        <td>5.17</td>
        <td>5.54</td>
        <td>182.20</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/fc29b93f483c898356685d3e0f269d16">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-626</a></td>
        <td>79.21</td>
        <td>6.26</td>
        <td>29.39</td>
        <td>4.27</td>
        <td>11.46</td>
        <td>38.53</td>
        <td>6.60</td>
        <td>6.51</td>
        <td>171.80</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/d397113cf40c96711424504d48f489f8">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-662</a></td>
        <td>80.83</td>
        <td>6.62</td>
        <td>70.45</td>
        <td>7.84</td>
        <td>23.57</td>
        <td>116.79</td>
        <td>6.67</td>
        <td>6.51</td>
        <td>163.70</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/b80decc99946291b788d7fa91cc25fae">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-676</a></td>
        <td>79.76</td>
        <td>6.76</td>
        <td>36.17</td>
        <td>4.60</td>
        <td>12.32</td>
        <td>46.65</td>
        <td>6.55</td>
        <td>6.47</td>
        <td>182.20</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/57fd10cb672224c1b00ff9b7394c6126">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-695</a></td>
        <td>79.53</td>
        <td>6.95</td>
        <td>29.38</td>
        <td>5.25</td>
        <td>12.33</td>
        <td>40.84</td>
        <td>8.75</td>
        <td>8.31</td>
        <td>160.70</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/daac73ed1b8764a2ea58dd723ad8deeb">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-834</a></td>
        <td>80.23</td>
        <td>8.34</td>
        <td>46.10</td>
        <td>5.53</td>
        <td>13.19</td>
        <td>42.65</td>
        <td>8.11</td>
        <td>8.68</td>
        <td>262.50</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/9ff9fef747889446025849db037fbd50">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-876</a></td>
        <td>81.67</td>
        <td>8.76</td>
        <td>47.83</td>
        <td>14.87</td>
        <td>41.51</td>
        <td>150.69</td>
        <td>19.05</td>
        <td>16.23</td>
        <td>317.90</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/4724c1b76c6cf238b0030e8a807c8e9b">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-1092</a></td>
        <td>80.39</td>
        <td>10.92</td>
        <td>42.21</td>
        <td>5.18</td>
        <td>17.18</td>
        <td>80.49</td>
        <td>7.11</td>
        <td>7.68</td>
        <td>232.50</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/cbb604a3615c5c750ef96956b9052914">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-1156</a></td>
        <td>80.61</td>
        <td>11.56</td>
        <td>43.03</td>
        <td>5.34</td>
        <td>17.92</td>
        <td>83.32</td>
        <td>7.31</td>
        <td>8.02</td>
        <td>260.50</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/bfe25fd8816984869328a24cdef0c5b1">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-1195</a></td>
        <td>80.63</td>
        <td>11.95</td>
        <td>45.49</td>
        <td>5.55</td>
        <td>18.40</td>
        <td>85.05</td>
        <td>7.95</td>
        <td>8.63</td>
        <td>259.10</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/75c3a51df7a35722849756b431205f13">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-1319</a></td>
        <td>81.38</td>
        <td>13.19</td>
        <td>72.44</td>
        <td>8.08</td>
        <td>19.88</td>
        <td>63.23</td>
        <td>14.14</td>
        <td>14.15</td>
        <td>300.40</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/12a1d57097023a7520456d10e9f127c8">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-1414</a></td>
        <td>81.22</td>
        <td>14.14</td>
        <td>79.39</td>
        <td>8.05</td>
        <td>21.49</td>
        <td>76.60</td>
        <td>12.34</td>
        <td>12.17</td>
        <td>251.90</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/ffeec3f0c9712be4807838c6c0a3e3b6">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-1549</a></td>
        <td>81.11</td>
        <td>15.49</td>
        <td>51.96</td>
        <td>6.37</td>
        <td>22.53</td>
        <td>112.33</td>
        <td>8.35</td>
        <td>9.51</td>
        <td>295.50</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/842fe47a276f6ff1b08e878c2dc61968">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-1772</a></td>
        <td>81.52</td>
        <td>17.72</td>
        <td>77.81</td>
        <td>7.67</td>
        <td>28.05</td>
        <td>128.57</td>
        <td>11.10</td>
        <td>12.29</td>
        <td>357.60</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/c86a0d7f3ee3c204a8f5514173dae68f">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-1822</a></td>
        <td>82.08</td>
        <td>18.22</td>
        <td>103.00</td>
        <td>11.80</td>
        <td>50.53</td>
        <td>298.63</td>
        <td>9.51</td>
        <td>12.11</td>
        <td>434.10</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/19d4376fca9d41cac492ece954d49c99">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-2354</a></td>
        <td>82.65</td>
        <td>23.54</td>
        <td>130.45</td>
        <td>20.94</td>
        <td>77.97</td>
        <td>397.44</td>
        <td>19.08</td>
        <td>21.13</td>
        <td>670.70</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/c07b888a68399440052c16b18eb514ab">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-2524</a></td>
        <td>82.04</td>
        <td>25.24</td>
        <td>76.66</td>
        <td>11.20</td>
        <td>35.08</td>
        <td>129.15</td>
        <td>18.71</td>
        <td>19.39</td>
        <td>504.90</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/c52a9a27de15e660b57d089cbfe69d70">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-2763</a></td>
        <td>82.42</td>
        <td>27.63</td>
        <td>87.34</td>
        <td>12.19</td>
        <td>38.15</td>
        <td>140.61</td>
        <td>19.96</td>
        <td>21.15</td>
        <td>599.60</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/a52781bb728a842433f3055845814564">tar</a></td>
    </tr>
    <tr>
        <td>D-Net-2883</a></td>
        <td>82.38</td>
        <td>28.83</td>
        <td>93.44</td>
        <td>12.25</td>
        <td>39.51</td>
        <td>146.81</td>
        <td>20.05</td>
        <td>21.54</td>
        <td>554.10</td>
        <td align="center"><a href="https://box.saas.huaweicloud.com/p/c9985754ae909bd63794628e44554809">tar</a></td>
    </tr>
</table>

### Part 2

| model_name | input | performance | Params(M) | FLOPS(G) | Acts(M) | bs1 | bs8 | bs64 | 224_1 | 224_8 | 512_1 | 512_8 | 720_1 | 720_8 | ptm_time_bs1 | ptm_time_bs8 | Download |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 2-_32_11-12-112-1121112 | 224 | 61.51  | 2.59  | 0.20  | 1.07  | 3.35  | 3.33  | 5.11  | 2.00  | 3.40  | 2.74  | 11.63  | 6.15  | 37.03  | 0.50  | 2.22  | [pth](https://box.saas.huaweicloud.com/p/a845f3a757e2d69ee66557088695c489) |
| 2-_32_2-11-112-1121112 | 224 | 62.06  | 2.62  | 0.23  | 1.27  | 3.23  | 3.25  | 5.30  | 1.87  | 3.58  | 2.74  | 11.64  | 6.59  | 41.28  | 0.53  | 2.42  | [pth](https://box.saas.huaweicloud.com/p/ccde393f82b728ac88ce635b0342b593) |
| 2-_32_2-11-1212-111112 | 224 | 64.49  | 3.16  | 0.29  | 1.38  | 2.52  | 3.11  | 5.94  | 2.04  | 3.62  | 2.92  | 12.88  | 7.15  | 45.50  | 0.60  | 2.65  | [pth](https://box.saas.huaweicloud.com/p/ac134eb01d9d03e4686fdcfc4229973d) |
| 031-_32_1-1-221-11121 | 224 | 66.92  | 3.94  | 0.39  | 1.28  | 3.12  | 2.69  | 5.92  | 2.05  | 3.53  | 2.84  | 12.08  | 6.33  | 38.83  | 0.66  | 2.63  | [pth](https://box.saas.huaweicloud.com/p/6d8af56c1346c2d2e4b722015d0daca3) |
| 011-_32_2-1-221-11121 | 224 | 67.71  | 4.37  | 0.38  | 1.56  | 3.52  | 3.38  | 6.48  | 2.05  | 3.80  | 2.99  | 13.06  | 7.19  | 45.45  | 0.75  | 3.08  | [pth](https://box.saas.huaweicloud.com/p/8a824763bf2dbddcdc169d684ac810ee) |
| 010-_64_1-211-2-11112 | 224 | 70.23  | 5.80  | 0.93  | 2.13  | 3.53  | 3.33  | 10.44  | 2.12  | 4.82  | 3.41  | 18.64  | 8.93  | 68.09  | 1.04  | 4.49  | [pth](https://box.saas.huaweicloud.com/p/6df263ff064633f7431836e0eb42ba8e) |
| 011-_32_2-211-2-111122 | 224 | 71.29  | 7.13  | 0.58  | 1.96  | 3.78  | 3.91  | 8.94  | 2.39  | 4.56  | 3.23  | 15.13  | 8.01  | 57.70  | 1.11  | 4.05  | [pth](https://box.saas.huaweicloud.com/p/8273ad94e98825d2ca090dd634d83dd0) |
| 011-_32_2-121-1121-11112111 | 224 | 71.27  | 7.59  | 0.66  | 2.10  | 5.14  | 4.12  | 9.48  | 2.69  | 5.19  | 3.65  | 16.70  | 8.69  | 62.01  | 1.19  | 4.39  | [pth](https://box.saas.huaweicloud.com/p/720d4970be23ce12633d10e95902de28) |
| 2-_64_2-211-2-11112 | 224 | 71.91  | 10.47  | 1.46  | 3.04  | 2.46  | 3.16  | 15.53  | 2.57  | 10.70  | 5.31  | 38.41  | 23.90  | 151.78  | 1.69  | 6.57  | [pth](https://box.saas.huaweicloud.com/p/cb4784b4eaafc6bdcc3ad27cf901f8aa)
| 031-_64_1-211-2-11112 | 224 | 74.18  | 10.56  | 1.66  | 3.07  | 3.26  | 3.48  | 16.38  | 2.65  | 7.43  | 4.25  | 27.33  | 12.33  | 99.13  | 1.71  | 6.83  | [pth](https://box.saas.huaweicloud.com/p/326a26be4719338f247c4d24e8ff9192) |
| 020-_64_1-211-2-11112 | 224 | 72.93  | 10.82  | 1.65  | 2.45  | 3.28  | 3.34  | 14.08  | 2.40  | 6.47  | 4.16  | 24.49  | 11.53  | 86.73  | 1.61  | 5.83  | [pth](https://box.saas.huaweicloud.com/p/fbd7d28bcf3ae7474c8e64c4c5fd76d7) |
| 10001-_48_41-1-221-11121 | 224 | 76.09  | 11.80  | 1.21  | 5.03  | 4.93  | 5.12  | 20.30  | 2.78  | 7.72  | 5.13  | 32.45  | 15.27  | 115.47  | 2.29  | 10.36  | [pth](https://box.saas.huaweicloud.com/p/a2907b2f55fb7bfd4546357e0800f36d) |
| 211-_32_41-211-121-11111121 | 224 | 76.29  | 12.85  | 1.42  | 4.19  | 4.92  | 4.46  | 20.04  | 3.36  | 11.30  | 5.97  | 39.53  | 18.11  | 145.97  | 2.20  | 8.88  | [pth](https://box.saas.huaweicloud.com/p/871db1caba1a33a323f220d03ef26078) |
| 333-a01_64_111-2111-211111-211 | 224 | 76.86  | 13.47  | 2.34  | 8.08  | 6.97  | 6.75  | 35.23  | 4.99  | 23.57  | 12.69  | 96.26  | 39.08  | 345.12  | 2.96  | 14.54  | [pth](https://box.saas.huaweicloud.com/p/2f51f75a0d229dee35e266b7f22fd8ff) |
| 31311-a02a12_64_211-2111-211111-211 | 224 | 78.04  | 14.17  | 2.32  | 8.46  | 7.89  | 8.19  | 40.55  | 5.62  | 22.02  | 12.19  | 96.61  | 40.86  | 359.24  | 3.47  | 18.11  | [pth](https://box.saas.huaweicloud.com/p/f16e87cc1dc9e00df28af6a8bbe62913) |
| 031-_64_1-1-221-11121 | 224 | 74.46  | 15.24  | 1.55  | 2.56  | 3.50  | 3.85  | 14.54  | 2.53  | 8.30  | 4.12  | 25.37  | 11.80  | 94.53  | 2.05  | 6.20  | [pth](https://box.saas.huaweicloud.com/p/7b765cd1e5828c17321bfb4192ba2044) |
| 421-_64_4-11-1212-111112 | 224 | 76.41  | 17.44  | 1.69  | 6.52  | 3.27  | 4.98  | 30.92  | 4.12  | 21.48  | 10.73  | 82.45  | 36.27  | 304.81  | 5.19  | 17.47  | [pth](https://box.saas.huaweicloud.com/p/e7592acdd5b5522400a4a3cae08ddfb0) |
| 031-_32_1-1211-112111112-22 | 224 | 75.21  | 19.13  | 1.57  | 2.47  | 5.11  | 5.93  | 15.81  | 3.15  | 8.89  | 4.77  | 25.03  | 12.55  | 93.93  | 2.39  | 6.35  | [pth](https://box.saas.huaweicloud.com/p/56edb1e4ad5fff05f83dc2eaf09749e3) |
| 031-_32_111-111121-1121111112-22 | 224 | 75.55  | 19.48  | 1.75  | 3.22  | 6.20  | 6.08  | 18.68  | 3.66  | 9.43  | 5.28  | 28.24  | 14.32  | 109.11  | 2.60  | 7.76  | [pth](https://box.saas.huaweicloud.com/p/5f2fafbc6be3fc1ba9197dc209c4bd20) |
| 031-_64_11-211-121-11111121 | 224 | 76.52  | 20.17  | 2.63  | 4.08  | 5.02  | 5.19  | 23.21  | 3.41  | 11.11  | 5.53  | 37.70  | 16.86  | 139.62  | 2.90  | 9.88  | [pth](https://box.saas.huaweicloud.com/p/34bd91f28f8ec90a5dfe457e9b933ac3) |
| 011-_64_21-211-121-11111121 | 224 | 76.87  | 21.42  | 2.61  | 4.69  | 5.10  | 5.65  | 25.34  | 3.64  | 11.89  | 5.91  | 44.06  | 19.24  | 160.38  | 3.16  | 11.12  | [pth](https://box.saas.huaweicloud.com/p/21de50409c639f06970008e25f785aae) |
| 031-_64_12-1111-11211112-2 | 224 | 78.16  | 24.42  | 4.56  | 6.10  | 4.81  | 6.93  | 38.58  | 3.49  | 14.01  | 8.75  | 66.67  | 29.43  | 248.20  | 3.77  | 15.15  | [pth](https://box.saas.huaweicloud.com/p/b1f0c01d13d10720612136984c67aab2) |
| 031-_64_1-1211-112111112-2 | 224 | 77.91  | 25.15  | 3.92  | 4.54  | 4.96  | 6.45  | 32.12  | 3.40  | 12.51  | 7.76  | 57.50  | 25.02  | 214.95  | 3.44  | 12.12  | [pth](https://box.saas.huaweicloud.com/p/19da2f85a3e510b01408a2f7da8a8965) |
| 031-_64_11-11211-112111112-2 | 224 | 78.24  | 25.31  | 4.24  | 5.29  | 4.98  | 6.54  | 35.19  | 3.56  | 13.27  | 8.37  | 62.53  | 27.06  | 232.30  | 3.65  | 13.63  | [pth](https://box.saas.huaweicloud.com/p/8f0e0f9be21cc93f8123f9a70ada18f8) |
| 10001-_64_4-111-11122-1111111111111112 | 224 | 78.80  | 28.45  | 2.31  | 6.80  | 8.18  | 8.24  | 33.83  | 5.03  | 12.20  | 8.00  | 52.07  | 21.43  | 180.97  | 4.36  | 15.48  | [pth](https://box.saas.huaweicloud.com/p/ee189b96e01c0090853c470f339778d6) |
| 031-_64_1-12112-111111112-2 | 224 | 78.96  | 28.47  | 5.01  | 5.47  | 4.78  | 7.41  | 39.57  | 3.60  | 14.71  | 9.04  | 70.00  | 30.32  | 263.07  | 3.96  | 14.76  | [pth](https://box.saas.huaweicloud.com/p/ed361356606718ef4fb49b444aa83078) |
| 02031-a02_64_112-1-11111111121112-1 | 224 | 79.53  | 29.38  | 6.90  | 10.44  | 9.12  | 10.92  | 55.25  | 5.31  | 18.96  | 12.21  | 94.28  | 40.77  | 349.24  | 5.15  | 23.45  | [pth](https://box.saas.huaweicloud.com/p/7d7d61027ad46a0390427b8a67741db4) |
| 011-_128_-111121111111-1211111112-11 | 224 | 79.21  | 29.39  | 6.23  | 7.45  | 7.22  | 9.66  | 48.53  | 4.20  | 17.58  | 11.28  | 86.41  | 38.45  | 327.18  | 4.69  | 20.20  | [pth](https://box.saas.huaweicloud.com/p/a19393971eb375435b7d5cb80faafc92) |
| 211-_64_4-11-1212-111112 | 224 | 78.28  | 30.76  | 2.72  | 5.52  | 4.50  | 6.50  | 30.58  | 4.31  | 20.67  | 10.58  | 78.84  | 34.65  | 283.53  | 4.28  | 13.36  | [pth](https://box.saas.huaweicloud.com/p/82296432ab977951214fee2d1d177b88) |
| 011-_128_111-21111-111112111121-111 | 224 | 79.76  | 36.17  | 6.73  | 7.88  | 6.90  | 10.41  | 52.64  | 4.70  | 20.58  | 12.35  | 96.16  | 46.64  | 395.22  | 5.55  | 21.88  | [pth](https://box.saas.huaweicloud.com/p/806ccbbf60677bca4657b224e2b9040f) |
| 011-_128_1-1111211111121-111111111121-11 | 224 | 80.32  | 39.50  | 9.53  | 9.84  | 8.59  | 13.37  | 68.68  | 5.28  | 25.96  | 15.58  | 125.84  | 62.52  | 534.45  | 6.29  | 28.21  | [pth](https://box.saas.huaweicloud.com/p/d6faca1fe84e56b98028c4f71a41cd1f) |
| 201-a01_64_4-121-1121-11112111 | 224 | 79.37  | 41.83  | 3.64  | 7.18  | 7.29  | 9.55  | 42.98  | 5.54  | 28.78  | 15.05  | 120.06  | 66.54  | 561.49  | 5.86  | 18.67  | [pth](https://box.saas.huaweicloud.com/p/ed23187c8b82312f887307214d400c16) |
| 011-_128_1-11111211211-111111112111-1 | 224 | 80.39  | 42.21  | 10.88  | 9.61  | 6.61  | 14.08  | 72.45  | 5.18  | 26.55  | 17.07  | 141.86  | 80.55  | 676.20  | 6.52  | 29.56  | [pth](https://box.saas.huaweicloud.com/p/b3496f2a36f575b9fced1268ede741ee) |
| 011-_128_1-111211111211-111111112111-1 | 224 | 80.61  | 43.03  | 11.52  | 10.21  | 7.03  | 14.89  | 76.31  | 5.34  | 27.80  | 17.85  | 148.11  | 83.31  | 699.94  | 6.76  | 31.42  | [pth](https://box.saas.huaweicloud.com/p/690732fd2979ca0326a16cc061c4ee07) |
| 10001-_64_4-111111111-211112111112-11111 | 224 | 80.92  | 44.00  | 5.33  | 10.94  | 8.90  | 12.83  | 63.61  | 5.82  | 29.65  | 13.83  | 111.33  | 46.99  | 419.47  | 6.90  | 26.26  | [pth](https://box.saas.huaweicloud.com/p/4d9878ffff3890cfd3077ea3f167f5fd) |
| 201-a01_64_41-121-11121-111112111 | 224 | 79.76  | 44.07  | 4.14  | 8.61  | 8.33  | 10.79  | 49.85  | 6.19  | 31.42  | 16.93  | 135.30  | 76.74  | 650.75  | 6.51  | 22.18  | [pth](https://box.saas.huaweicloud.com/p/bf4068ff951785e25d484f13e6cc07af) |
| 011-_128_1-1111121111211-11111111112111-1 | 224 | 80.63  | 45.49  | 11.90  | 10.51  | 8.14  | 15.39  | 79.23  | 5.78  | 28.85  | 18.57  | 152.05  | 84.91  | 717.28  | 7.07  | 32.50  | [pth](https://box.saas.huaweicloud.com/p/25b0aa6643d3761fa51c842b69548627) |
| 011-_64_211-2111-21111111111111111111111-211 | 224 | 80.23  | 46.10  | 8.30  | 8.46  | 8.58  | 12.26  | 61.03  | 5.55  | 22.94  | 13.34  | 101.35  | 42.82  | 399.46  | 6.58  | 25.46  | [pth](https://box.saas.huaweicloud.com/p/adfecfc4af4fefb7dda09899f6c72d23) |
| 32341-a02c12_64_111-2111-21111111111111111111111-211 | 224 | 81.67  | 47.83  | 8.67  | 23.51  | 16.48  | 23.94  | 128.54  | 14.97  | 91.15  | 41.54  | 379.11  | 150.41  | 1366.67  | 9.50  | 43.72  | [pth](https://box.saas.huaweicloud.com/p/9f7b25b91ddd8d66b96c7ed97ff6a4fe) |
| 211-_64_41-211-121-11111121 | 224 | 80.41  | 49.29  | 5.68  | 8.38  | 5.43  | 10.82  | 53.08  | 6.17  | 36.73  | 18.62  | 154.45  | 87.25  | 714.54  | 7.02  | 23.36  | [pth](https://box.saas.huaweicloud.com/p/9e633231e4c1e6d19147e348d01a18f3) |
| 011-_128_1-111121111211111-1111111121111-1 | 224 | 81.27  | 51.96  | 15.44  | 12.17  | 7.78  | 18.77  | 95.30  | 6.28  | 34.92  | 22.48  | 194.34  | 112.36  | 944.00  | 8.20  | 39.98 | [pth](https://box.saas.huaweicloud.com/p/2b317dae184b2a07a32867119fd69c50) |
| 23311-a02c12_64_211-2111-211111-211 | 224 | 81.66  | 66.87  | 11.01  | 12.60  | 10.23  | 19.38  | 102.58  | 11.84  | 79.11  | 41.04  | 360.70  | 196.96  | 1713.64  | 10.42  | 39.89  | [pth](https://box.saas.huaweicloud.com/p/404e6321b1020f6db5d5f31df592d2bb) |
| 211-_64_41-121-11121-111112111 | 224 | 80.83  | 70.45  | 6.58  | 8.81  | 6.74  | 12.96  | 59.13  | 7.74  | 48.90  | 23.48  | 190.71  | 116.54  | 962.28  | 9.00  | 24.70  | [pth](https://box.saas.huaweicloud.com/p/431407690420676795584b41f377de63) |
| 02031-a02_64_111-2111-21111111111111111111111-211 | 224 | 81.38  | 72.44  | 13.13  | 14.58  | 15.52  | 19.53  | 94.60  | 8.20  | 33.68  | 20.03  | 150.70  | 63.20  | 580.97  | 10.13  | 38.02  | [pth](https://box.saas.huaweicloud.com/p/84d1d3617c400af6bbd2cb9685648144) |
| 02031-a02_64_1121-11111111111111111111111111-211111121111-1 | 224 | 82.04  | 76.66  | 25.12  | 29.83  | 17.32  | 32.57  | 172.00  | 11.33  | 52.20  | 34.95  | 284.90  | 129.17  | 1120.81  | 13.90  | 73.14  | [pth](https://box.saas.huaweicloud.com/p/af3e366e29482cac071c7f0d49aa90bf) |
| 011-_128_2-111111111111111111-121111111111121111111-2 | 224 | 81.52  | 77.81  | 17.66  | 14.53  | 11.25  | 22.59  | 114.93  | 7.67  | 42.09  | 27.95  | 223.36  | 128.39  | 1080.82  | 11.35  | 48.58  | [pth](https://box.saas.huaweicloud.com/p/057c0152071d3ccf4a9e3490d360a1d7) |
| 02031-a02_64_1-111111211111-111121121111-2 | 224 | 81.22  | 79.39  | 14.08  | 12.82  | 12.00  | 19.53  | 92.20  | 8.02  | 36.02  | 21.53  | 166.44  | 76.58  | 660.04  | 10.36  | 37.02  | [pth](https://box.saas.huaweicloud.com/p/97eaca8b0072565b6d23649a22dfb344) |
| 02031-a02_64_1121-111111111111111111111111111-21111111211111-1 | 224 | 82.42  | 87.34  | 27.51  | 31.34  | 21.72  | 35.34  | 185.75  | 11.97  | 56.46  | 38.01  | 308.30  | 140.38  | 1219.08  | 15.25  | 78.39  | [pth](https://box.saas.huaweicloud.com/p/cf4378ea80f370768a4e4592b2264f52) |
| 02031-a02_64_1121-111111111111111111111111111-21111112111111-1 | 224 | 82.38  | 93.44  | 28.70  | 31.64  | 18.79  | 36.44  | 191.20  | 12.21  | 58.64  | 39.37  | 319.61  | 146.69  | 1269.18  | 15.90  | 80.45  | [pth](https://box.saas.huaweicloud.com/p/86da19d8308876d558618e0af4720ae1) |
| 211-_64_411-2111-21111111111111111111111-211 | 224 | 82.08  | 103.00  | 18.16  | 15.91  | 10.50  | 25.20  | 131.92  | 11.61  | 75.93  | 50.39  | 418.51  | 298.10  | 2543.09  | 14.29  | 52.45  | [pth](https://box.saas.huaweicloud.com/p/257a9c954231701b0695cb849f90587b) |
| 23311-a02c12_64_211-2111-21111111111111111111111-211 | 224 | 82.65  | 130.45  | 23.46  | 19.42  | 18.00  | 36.97  | 191.89  | 20.87  | 136.62  | 77.75  | 664.57  | 396.85  | 3414.92  | 18.66  | 68.88  | [pth](https://box.saas.huaweicloud.com/p/42830031265b581ebeaebecfc71dbe08) |

## Reference

The compressed package of each model contains the model and inference sample code. If you have any questions, submit the issue in a timely manner. We will reply to you in a timely manner.
