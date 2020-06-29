# model zoo

## Image Classification on Cifar10

| Model | Model Size(M) | Flops(M) | Top-1 ACC | Inference Time(μs) | Inference Device | Download |
|---|---|---|---|---|---|:--:|
| CARS-A | 7.72 | 469 | 95.923477 | 51.28 | V100 | [tar](http://www.noahlab.com.hk/opensource/vega/model_zoo/cars/cars_a.zip) |
| CARS-B | 8.45 | 548 | 96.584535 | 69.00 | v100 | [tar](http://www.noahlab.com.hk/opensource/vega/model_zoo/cars/cars_b.zip) |
| CARS-C | 9.32 | 620 | 96.744791 | 71.62 | v100 | [tar](http://www.noahlab.com.hk/opensource/vega/model_zoo/cars/cars_c.zip) |
| CARS-D | 10.5 | 729 | 97.055288 | 82.72 | v100 | [tar](http://www.noahlab.com.hk/opensource/vega/model_zoo/cars/cars_d.zip) |
| CARS-E | 11.3 | 786 | 97.245592 | 88.98 | v100 | [tar](http://www.noahlab.com.hk/opensource/vega/model_zoo/cars/cars_e.zip) |
| CARS-F | 16.7 | 1234 | 97.295673 | 244.07 | v100 | [tar](http://www.noahlab.com.hk/opensource/vega/model_zoo/cars/cars_f.zip) |
| CARS-G | 19.1 | 1439 | 97.375801 | 391.20 | v100 | [tar](http://www.noahlab.com.hk/opensource/vega/model_zoo/cars/cars_g.zip) |
| CARS-H | 19.6 | 1464 | 97.415865 | 398.19 | v100 | [tar](http://www.noahlab.com.hk/opensource/vega/model_zoo/cars/cars_h.zip) |
| CARS-I | 19.7 | 1456 | 97.425881 | 398.88 | P100 | [tar](http://www.noahlab.com.hk/opensource/vega/model_zoo/cars/cars_i.zip) |

## Image Classification on ImageNet

| Model | Model Size(M) | Flops(G) | Top-1 ACC | Inference Time(s) | Download |
|---|---|---|---|---|:--:|
| EfficientNet:B8:672 | 88 | 63 | 85.7 | 0.98s/iters | [tar](http://www.noahlab.com.hk/opensource/vega/model_zoo/effecientnet/efficientnet.tar.gz) |
| EfficientNet:B8:832 | 88 | 97 | 85.8 | 1.49s/iters | [tar](http://www.noahlab.com.hk/opensource/vega/model_zoo/effecientnet/efficientnet.tar.gz) |

## Detection on COCO-minival

| Model | Model Size(M) | Flops(G) | mAP | Inference Time(ms) | Inference Device | Download |
|---|---|---|---|---|---|:--:|
| SM-NAS:E0 | 16.23 | 22.01 | 27.11 | 24.56 | V100 | [tar](http://www.noahlab.com.hk/opensource/vega/model_zoo/sm_nas/E0.tar.gz) |
| SM-NAS:E1 | 37.83 | 64.72 | 34.20 | 32.07 | V100 | [tar](http://www.noahlab.com.hk/opensource/vega/model_zoo/sm_nas/E1.tar.gz) |
| SM-NAS:E2 | 33.02 | 77.04 | 40.04 | 39.50 | V100 | [tar](http://www.noahlab.com.hk/opensource/vega/model_zoo/sm_nas/E2.tar.gz) |
| SM-NAS:E3 | 52.05 | 116.22 | 42.68 | 50.71 | V100 | [tar](http://www.noahlab.com.hk/opensource/vega/model_zoo/sm_nas/E3.tar.gz) |
| SM-NAS:E4 | 92 | 115.51 | 43.89 | 80.22 | V100 | [tar](http://www.noahlab.com.hk/opensource/vega/model_zoo/sm_nas/E4.tar.gz) |
| SM-NAS:E5 | 90.47 | 249.14 | 46.05 | 108.07 | V100 | [tar](http://www.noahlab.com.hk/opensource/vega/model_zoo/sm_nas/E5.tar.gz) |

## Detection on CULane

| Model | Flops(G) | F1 Score | Inference Time(ms) | Inference Device | Download |
|---|---|---|---|---|:--:|
| AutoLane | 86.5 | 74.8 | 44.94 | V100 | [tar](http://www.noahlab.com.hk/opensource/vega/model_zoo/auto_lane/auto_lane.tar.gz) |

## Detection on ECP

| Model | Model Size(M) | Flops(G) | F1 Score | Inference Time(ms) | Inference Device | Download |
|---|---|---|---|---|---|:--:|
| SP-NAS | 115.9 | 984.2 | 0.024 | 322.00 | V100 | [tar](http://www.noahlab.com.hk/opensource/vega/model_zoo/sp_nas/sp_nas.tar.gz) |

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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/esr_ea/esr_ea_1.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/esr_ea/esr_ea_2.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/esr_ea/esr_ea_3.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/esr_ea/esr_ea_4.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/sr_ea/A.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/sr_ea/B.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/sr_ea/C.zip">tar</a></td>
    </tr>
<table>

## Segmentation on VOC2012

| Model | Model Size(M) | Flops(G) | KParams | mIOU | Download |
|---|---|---|---|---|:--:|
| Adelaide | 10.6 | 0.5784 | 3822.14 | 0.7602 |[tar](http://www.noahlab.com.hk/opensource/vega/model_zoo/adelaide/adelaide.zip) |

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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-21.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-24.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-30.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-39.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-40.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-59.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-94.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-124.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-147.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-156.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-159.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-166.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-167.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-172.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-177.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-234.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-263.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-264.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-275.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-367.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-394.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-504.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-538.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-572.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-626.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-662.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-676.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-695.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-834.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-876.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-1092.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-1156.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-1195.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-1319.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-1414.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-1549.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-1772.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-1822.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-2354.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-2524.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-2763.zip">tar</a></td>
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
        <td align="center"><a href="http://www.noahlab.com.hk/opensource/vega/model_zoo/DNet/D-Net-2883.zip">tar</a></td>
    </tr>
</table>

## Reference

每个模型的压缩包中包含了模型和推理示例代码，若大家在使用中有任何问题，请及时提交issue，我们会及时答复。
