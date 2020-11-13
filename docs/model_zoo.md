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

## Reference

The compressed package of each model contains the model and inference sample code. If you have any questions, submit the issue in a timely manner. We will reply to you in a timely manner.
