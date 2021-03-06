---
layout: post
title:  "Bone Segmentation Using Image Registration with pyANTs"
date:   2021-11-09 16:07:59 +0200
categories: 
---
In this post I'll walk you through a python example showing how to use image registration to remove the bones from a head CT scan.
The registration will be done between two head scans from two different 3D modalities - MRI and CT using the ANTs library.

A jupyter notebook with the entire code of this post is available [here](https://github.com/claraherscu/claraherscu.github.io/blob/4a9b4a95419ad276fb541ae0dadfd33b7585b9c1/assets/jupyter_notebooks/Bone%20Segmentation%20Using%20Image%20Registration%20with%20pyANTs.ipynb).

## Introduction

[Image Registration](https://en.wikipedia.org/wiki/Medical_image_computing#Registration) is the process of aligning multiple images to the 
same coordinate system. This can be useful for tasks that involve local comparison between images 
like tracking changes across multiple images, for example tracking tumor growth along time, identifying new objects in satellite images of the same area, etc.

Another cool use of registration is [atlas-based segmentation](https://en.wikipedia.org/wiki/Medical_image_computing#Atlases). 
Say you have a segmentation task for which you have a solution over one image, you can use the segmentation masks you have and apply them to other images if you align them all.

Specifically, in this post I'll use ANTs to register a head CT scan to an MRI scan with a corresponding bone mask, to get a masked version of the CT scan.

## The Data
In this post I'll use two types of head scans: 
1. Head MRI - [MNI305](http://nist.mni.mcgill.ca/mni-average-brain-305-mri/). 
   This is a common atlas used in the field, and it also has various segmentation masks we can use.
   This will be our static template image, other images will be registered to it and can then use its masks.
2. Head CT - from the [CQ500 dataset](http://headctstudy.qure.ai/dataset).
   These are the unsegmented images we are going to register to our template.

Even though each of these scans is a 3D image volume of a human head, they have quite different characteristics in terms of shape and pixel values.

```python
import nibabel as nib
import numpy as np

def get_data_from_nii_path(nii_data_path: str) -> np.ndarray:
    series_data = nib.load(nii_data_path).get_data()
    # reorder data to (slices, height, width)
    series_data = np.transpose(series_data, [2, 1, 0])
    return series_data

mri_series_path = '/path/to/mri/data/'
ct_series_path = '/path/to/ct/data/'

mri_series = get_data_from_nii_path(mri_series_path)
ct_series = get_data_from_nii_path(ct_series_path)
```

<center><img src="/assets/bone_segmentation_assets/mri_example_slice.png" alt="Example slice from an MRI scan" height="400"/>  <img src="/assets/bone_segmentation_assets/ct_example_slice_windowed.png" alt="Example slice from a CT scan" height="400"/></center>
<center>Example slice from a head MRI scan (left) and a CT scan (right)<br/><br/></center>

```python
print(mri_series.shape)
>>> (156, 220, 172)
print(ct_series.shape)
>>> (239, 512, 512)

print(f'MRI: minimum value={mri_series.min()}, maximum value={mri_series.max()}')
>>> MRI: minimum value=-2.695594434759768e-05, maximum value=110.81832135075183
print(f'CT: minimum value={ct_series.min()}, maximum value={ct_series.max()}')
>>> CT: minimum value=-3023, maximum value=3071
```

Like I mentioned, the atlas MRI comes with a soft matter (anything not a bone) segmentation mask. Here is a visualization of the mri with its segmentation.

<center><img src="/assets/bone_segmentation_assets/mri_masked.gif" alt="MRI with mask" height="500"/></center>
<center>MRI with corresponding mask<br/><br/></center>

## A (very very) naiive approach

What would happen if we try to use the MRI mask on one of the CT scans without registering them first? 
Lets give it a try:

<center><img src="/assets/bone_segmentation_assets/ct_masked_no_registration.gif" alt="CT with naiively reshaped mask" height="500"/></center>
<center>CT with naiively reshaped mask<br/><br/></center>

As expected, this could never work. But, after we align the CT scan to the same coordinates of the template, the mask will fit perfectly! 

## Image Registration

Image registration is the well researched problem of *transforming two images into the same coordinate system*.
Specifically, one of the images defines the target coordinate system and will be called the *fixed image* or *static image* 
(it will not move or change during this process), and the other, *moving image* will be registered to it.

There are a few decisions we have to make when approaching a registration task. 
1. How much are we willing to distort the moving image? This will determine the degrees of freedom we want in our registration model
2. What similarity metric fits the images we are using?
3. How do we want to perform interpolation for pixels? 
4. Do we want to use the image intensity directly (intensity-based registration) or extract features from the image and use them to perform the registration (feature-based registration)?

In this example I'll perform an **intensity-based** optimization, and due to the different characteristics of the images, 
I'll allow a **big amount of distortion**, therefore using the ants `'SyN'` transformation. 
For a less accurate registration that would not distort the moving image, I could use the ants `'Affine'` or `'TRSAA'` transformations 
(for more details about types of transformations please check out the documentation notes [here](https://antspy.readthedocs.io/en/latest/registration.html)).  

As for the possible similarity metrics, for most registration tasks, usually either 
[Mutual Information](https://en.wikipedia.org/wiki/Mutual_information) or [Cross Correlation](https://en.wikipedia.org/wiki/Cross-correlation) will be enough. 
In general, Cross Correlation works well for intra-modality registration, and Mutual Information works well for both intra- and inter-modality registration. 
Here, since the example is inter-modality, I'll use **Mutual Information**.

Before we start, let???s look at the two unregistered 3D scans:
<center><img src="/assets/bone_segmentation_assets/before_registration.gif" alt="slices from MRI and CT before registration" height="500"/></center>
<center>Slices from the CT and the MRI side by side before registration<br/><br/></center>


**OK, so let's do this!**

### Step 0: numpy to ants (and back)

First, I'll define some basic functions for converting numpy data into ants objects and vice versa.

```python
import ants

def get_ants_from_numpy(arr: np.ndarray) -> ants.ANTsImage:
    arr = np.copy(np.transpose(arr, [1, 2, 0]))
    arr = arr.astype(np.float32)
    return ants.from_numpy(arr)


def get_numpy_from_ants(a: ants.ANTsImage) -> np.ndarray:
    arr = a.numpy()
    arr = np.transpose(arr, [2, 0, 1])
    arr = arr.astype(np.int)
    return arr
```

### Step 1: Get transformation parameters

The registration process that finds the transformation parameters to align `moving_series` to `static_series` 
is done by [ants.registration](https://antspy.readthedocs.io/en/latest/registration.html).

 ```python
 DEFAULT_REGISTRATION_ITERATIONS = (10, 10, 10)
 MI_METRIC = 'mattes'
 SYN_TRANSFORM = 'SyN'
 
 def get_transformation_params(reference_ants: ants.ANTsImage, moving_ants: ants.ANTsImage, registration_iterations: tuple = DEFAULT_REGISTRATION_ITERATIONS,
                               transform_type: str = SYN_TRANSFORM, similarity_metric: str = MI_METRIC) -> dict:
     return ants.registration(fixed=reference_ants, moving=moving_ants, type_of_transform=transform_type, reg_iterations=registration_iterations, aff_metric=similarity_metric)
 ```

We have just calculated the transformation parameters, allowing us to map each pixel in `moving_series` to it's new location where it will be aligned with `static_series`.

Actually `ants.registration` does more than that and also returns a bunch of additional useful things. It returns a dictionary, containing:

| Key | Value meaning |
| ----------- | ----------- |
| `'fwdtransforms'` | The transformation parameters to register `moving_series` to `static_series` |
| `'invtransforms'` | The inverse transformation parameters, that is, to register `static_series` to `moving_series` |
| `'warpedmovout'` | The `moving_series` after applying the transformation we have just calculated. That is, `moving_series` warped to be registered to `static_series` |
| `'warpedfixout'` | The `static_series` after applying the inverse transformation to it. That is, `static_series` warped to be registered to `moving_series` |

### Step 2: Apply transformation

To apply the transformation parameters to `moving_series`, I'll use [ants.apply_transforms](https://antspy.readthedocs.io/en/latest/registration.html?highlight=apply_transform#ants.apply_transforms). 

```python
INTERPOLATION_TYPE = 'bSpline'

def apply_transformation(reference_ants: ants.ANTsImage, moving_ants: ants.ANTsImage, transformation: dict, interpolator: str = INTERPOLATION_TYPE) -> ants.ANTsImage:
    return ants.apply_transforms(fixed=reference_ants, moving=moving_ants,
                                 transformlist=transformation['fwdtransforms'], interpolator=interpolator)
```

> **Note**: I could take the 'warpedmovout' from the returned dictionary of ants.registration instead of doing this as a separate step, but I like to control the interpolation explicitly.
> 
> This is specifically useful when you want to use the same parameters to register something in addition to the `moving_series`. 
> For example, often we'd want to use the same parameters to register a segmentation mask for the `moving_series` to the same coordinate system. 
> In that case, we'd use `'nearestNeighbor'` or `'genericLabel'` interpolator.

### Step 3: Tying it all together

```python
def register_series(moving_series_array: np.ndarray, reference_series_array: np.ndarray,
                    registration_iterations: tuple = DEFAULT_REGISTRATION_ITERATIONS, 
                    transform_type: str = SYN_TRANSFORM, 
                    similarity_metric: str = MI_METRIC) -> (np.ndarray, dict):
    # convert both scans to ants objects
    moving_series_ants = get_ants_from_numpy(moving_series_array)
    reference_series_ants = get_ants_from_numpy(reference_series_array)

    # get transformation parameters
    transformation = get_transformation_params(reference_series_ants, moving_series_ants, 
                                               registration_iterations=registration_iterations, 
                                               transform_type=transform_type, 
                                               similarity_metric=similarity_metric)

    # apply the transformation
    registered_ants_series = apply_transformation(reference_series_ants, moving_series_ants, 
                                                  transformation, 
                                                  interpolator=INTERPOLATION_TYPE)

    # retrieve registered numpy array
    registered_moving_series = get_numpy_from_ants(registered_ants_series)
    return registered_moving_series, transformation

registered_moving_series, transformation = register_series(moving_series, static_series)
```

After this registration process, `registered_moving_series` above, is a new version of `moving_series`, after it was registered to `static_series`.
Let's look at both 3d images again, this time registered!

<center><img src="/assets/bone_segmentation_assets/after_registration.gif" alt="slices from MRI and CT after registration" height="500"/></center>
<center>Slices from the CT and the MRI side by side after registration<br/><br/></center>

Note that this means it's shape has changed to match the shape of `static_series`:
```python
print(f'Shape comparison')
print(f'Moving image shape before registration: {moving_series.shape}')
print(f'Target image shape before registration: {static_series.shape}')
print(f'Moving image shape after registration: {registered_moving_series.shape}')

>>> Shape comparison 
>>> Moving image shape before registration: (239, 512, 512)
>>> Target image shape before registration: (156, 220, 172)
>>> Moving image shape after registration: (156, 220, 172)
```

### Step 4: Using the MNI Segmentation Mask

Once we've registered the CT to the MRI, it's now trivial to use the MRI mask over the CT in that space.

<center><img src="/assets/bone_segmentation_assets/ct_mask_mni_space.png" alt="Example slice from a CT scan in the MNI space with brain mask" height="400"/></center>
<center>Example slice from the CT scan after registration to the MRI, with the brain mask<br/><br/></center>

We might also use the MRI mask in the CT original coordinate system. To do that, we simply apply the inverse transformation on the segmentation mask.
```python
segmentation_mask_ants = get_ants_from_numpy(mask)

def apply_inverse_transformation(reference_ants: ants.ANTsImage, moving_ants: ants.ANTsImage, transformation: dict, interpolator: str = INTERPOLATION_TYPE) -> ants.ANTsImage:
    return ants.apply_transforms(fixed=reference_ants, moving=moving_ants,
                                 transformlist=transformation['invtransforms'], interpolator=interpolator)

mask_transformed_ants = apply_inverse_transformation(moving_series_ants, 
                                                     segmentation_mask_ants, 
                                                     transformation, 
                                                     interpolator='nearestNeighbor')
mask_transformed = get_numpy_from_ants(mask_transformed_ants)
```

<center><img src="/assets/bone_segmentation_assets/ct_masked_after_registration.gif" alt="CT masked after registration" height="500"/></center>
<center>CT scan with the mask after registration - In original ct coordinate system<br/><br/></center>

> **Note**: The bone removal isn't perfect because CT and MRI are essentially different and it's hard to get a perfect registration between them, 
> so we should tweak the parameters to fit this specific problem perfectly (for example I used a very small number of iterations in the optimization process). 
  

## Summary and Pro Tips

I've walked you through a working example of using python with ANTs library in order to register a 3D CT image to a 3D MRI image.

The tricky part in making a registration process actually work on real data will often be choosing the correct parameters for your specific problem:
- Similarity metric
- Type of transformation 
- Iterations (tradeoff of runtime and accuracy)
- Interpolation technique

***Pro tip #1:*** In many cases, it's enough to use low resolution images in order to get the transformation parameters right, 
and it will significantly speed up the optimization process. 
Therefore, you can drastically downsample your images before getting the transformation parameters, and apply the transformation you got on the full resolution image for improved runtime.

***Pro tip #2:*** `ants.registration` function also takes an optional argument `mask`, which is a binary mask over the `static_image` defining the areas of the image the registration will focus on. 
It can be useful to supply such a mask to disregard areas in the image with no relevant information for the registration task (e.g. background objects). 
