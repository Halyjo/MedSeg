"""
Classes and function to visualize the data and results.
"""
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from utils import load_volume


class ImageViewer():
    _focus_options = ['liver', 'lesion']

    def __init__(self, src_path, dst_path=None):
        """
            Arguments
            ---------
                src_path : str
                    Path to directory containing two folders.
                    - slices/
                        Containing slice_XXXXX.npy files (XXXXX is a number)
                    - labels/
                        Containing segmentation_XXXXX.npy files (XXXXX is a number)
                [dst_path] : str
                    Default: "./"
                    If specified, path to store images while viewing.
        """
        self.src_path = src_path
        self.dst_path = dst_path if dst_path is not None else "./"

    def view_img(self, idx=1, focus='liver', singleview=False):
        """
        Function to view image with labeled segmentation map 
        and prediction if available.

            Arguments
            ---------
                [idx] : int
                    Default: 0
                    Index of image to view.
                [focus] : ['liver', 'lesion']
                    Default: 'liver'
                    Which image segmentation map to get path of. 

        """
        ## Get volumes list based on path and index.
        assert isinstance(idx, int), "idx must be an integer."
        assert focus in self._focus_options, f"focus must be among: {self._focus_options}."
        assert singleview in [True, False], "singleview must be boolean value."

        self.idx = idx
        self.focus = focus
        self.singleview = singleview
        self._remove_keymap_conflicts({'left', 'right'})
        if self.singleview:
            fig, ax = plt.subplots(nrows=1, ncols=3)
        else:
            fig, ax = plt.subplots(nrows=5, ncols=6)
        ax = ax.flatten()
        if self.singleview:
            fig = self._fill_one_index(fig, replace=False)
        else:
            fig = self._fill_canvas(fig, replace=False)
            
        fig.canvas.mpl_connect('key_press_event', self._process_key)
        plt.show()

    def get_paths(self, idx=1, focus='liver'):
        """ Get paths of volume, label and prediction. View docstring for __init__
        for assumtions about file structure.

            Arguments
            ---------
                [idx] : int
                    Default: 0
                    Index of image to view.
                [focus] : ['liver', 'lesion']
                    Default: 'liver'
                    Which image segmentation map to get path of. 

            Returns
            -------
                paths : dict
                    keys: ['slice', 'lab', 'pred']
                    if the paths do not exist 
                [focus] : str
                    Default: "liver"
                    options: ['liver', 'lesion']
                    Which labels and prediction to view. Segmented livers or segmented lesions.
        """
        keys = ['slice', 'lab', 'pred']
        paths = [
            os.path.join(self.src_path, "slices/slice_{:05}.npy".format(idx)),
            os.path.join(self.src_path, "labels_{}/segmentation_{:05}.npy".format(focus, idx)),
            os.path.join(self.src_path, "pred_{}/prediction_{:05}.npy".format(focus, idx)),
        ]
        path_dict = {}
        for k, p in zip(keys, paths):
            if os.path.exists(p):
                print(f"{k}: {p}", "added.")
                path_dict.update({k: p})
            else:
                print(p, "does not exist.")

        msg = "Volume or label location not found. Paths tried: \n{}".format(paths)
        assert len(path_dict) >= 2, msg
        if len(path_dict) == 2:
            print("Warning: Prediction for volume {} is not available".format(idx))
        return path_dict

    def _fill_canvas(self, fig, replace=True):
        ax = fig.axes
        start = self.idx
        end = self.idx + 10
        slice_list = []
        names = []
        for i in range(start, end):
            paths = self.get_paths(i, self.focus)
            slice_list.extend([np.load(p) for p in paths.values()])
            names.extend(list(paths.keys()))
        for i, img in enumerate(slice_list):
            if i<6:
                ax[i].set_title(names[i])
            if replace:
                ax[i].images[0].set_array(img)
                # print(np.unique(img, return_counts=True)[1])
            else:
                # img = img/np.max(img)
                ax[i].imshow(img, cmap='magma')
            ax[i].axis('off')
        return fig

    def _fill_one_index(self, fig, replace=True):
        ax = fig.axes
        slice_list = []
        names = []
        paths = self.get_paths(self.idx, self.focus)
        slice_list.extend([np.load(p) for p in paths.values()])
        names.extend(list(paths.keys()))
        for i, img in enumerate(slice_list):
            ax[i].set_title(names[i])
            if replace:
                ax[i].images[0].set_array(img)
            else:
                ax[i].imshow(img, cmap='magma')
            ax[i].axis('off')
        return fig

    def _remove_keymap_conflicts(self, new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def _process_key(self, event):
        fig = event.canvas.figure
        if event.key == 'left':
            self._previous_bunsh(fig)
        elif event.key == 'right':
            self._next_bunsh(fig)
        elif event.key == 's':
            plt.savefig("fig{:03}.png".format(np.random.randint(0, 1000)), dpi=300)
        fig.canvas.draw()

    def _previous_bunsh(self, fig):
        if self.singleview:
            self.idx -= 1
            self._fill_one_index(fig)
        else:
            self.idx -= 10
            self._fill_canvas(fig)
        
    def _next_bunsh(self, fig):
        if self.singleview:
            self.idx += 1
            self._fill_one_index(fig)
        else:
            self.idx += 10
            self._fill_canvas(fig)


class VolumeViewer():
    _focus_options = ['liver', 'lesion']
    # _mode_options = ['train', 'test']
    def __init__(self, src_path, dst_path=None):
        """
            Arguments
            ---------
                src_path : str
                    Path to directory containing two folders.
                    - volumes/
                        Containing volume_XX.nii files (XX is a number)
                    - labels/
                        Containing segmentation_XX.nii files (XX is a number)
                [dst_path] : str
                    Default: "./"
                    If specified, path to store images while viewing.
        """
        self.src_path = src_path
        self.dst_path = dst_path if dst_path is not None else "./"

    def view_volume(self, idx, focus):
        """
        Function to view one or multiple volumes simultaniously.

            Arguments
            --------
                idx : int
                    Index of volume to view. Assumes there exists

        """
        ## Get volumes list based on path and index.
        assert isinstance(idx, int), "idx must be an integer."
        assert focus in self._focus_options, f"focus must be among: {self._focus_options}."
        paths = self.get_paths(idx, focus)
        volslist = [load_volume(p) for p in paths.values()]
        names = list(paths.keys())

        self._remove_keymap_conflicts({'j', 'k', 's'})
        
        fig, ax = plt.subplots(ncols=len(volslist))
        for i in range(len(volslist)):
            print(volslist[i].shape)
            ax[i].volume = volslist[i]
            ax[i].index = volslist[i].shape[0] // 2
            ax[i].imshow(volslist[i][ax[i].index],
                         cmap='magma')
            ax[i].set_title(names[i])
        fig.canvas.mpl_connect('key_press_event', self._process_key)
        plt.show()

    def get_paths(self, idx=0, focus='liver'):
        """ Get paths of volume, label and prediction. View docstring for __init__
        for assumtions about file structure.

            Returns
            -------
                paths : dict
                    keys: ['vol', 'lab', 'pred']
                    if the paths do not exist 
                idx : int
                    Index of volume to view (with labels and predictions).
                [focus] : str
                    Default: "liver"
                    options: ['liver', 'lesion']
                    Which labels and prediction to view. Segmented livers or segmented lesions.
        """
        keys = ['vol', 'lab', 'pred']
        paths = [
            os.path.join(self.src_path, "volumes/volume-{:02}.nii".format(idx)),
            os.path.join(self.src_path, "labels_{}/segmentation-{:02}.nii.gz".format(focus, idx)),
            os.path.join(self.src_path, "pred_{}/prediction_{:02}.nii".format(focus, idx)),
        ]
        path_dict = {}
        for k, p in zip(keys, paths):
            if os.path.exists(p):
                print(f"{k}: {p}", "added.")
                path_dict.update({k: p})
            else:
                print(p, "does not exist.")

        msg = "Volume or label location not found. Paths tried: \n{}".format(paths)
        assert len(path_dict) >= 2, msg
        if len(path_dict) == 2:
            print("Warning: Prediction for volume {} is not available".format(idx))
        return path_dict

    def _remove_keymap_conflicts(self, new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def _process_key(self, event):
        fig = event.canvas.figure
        # ax = fig.axes[0]
        for ax in fig.axes:
            if event.key == 'j':
                self._previous_slice(ax)
            elif event.key == 'k':
                self._next_slice(ax)
            elif event.key == 's':
                plt.savefig("fig{:02}.png".format(np.random.randint(0, 100)), dpi=300)
        fig.canvas.draw()

    def _previous_slice(self, ax):
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
        ax.images[0].set_array(volume[ax.index])

    def _next_slice(self, ax):
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("View LiTS training data.")
    parser.add_argument("--idx", type=int, default=0,
                        help="Index of training image to view.")
    parser.add_argument("--focus", type=str, default="liver",
                        help="'liver' or 'lesion'.")
    parser.add_argument("--mode", type=str, default='train',
                        choices=('train', 'test'),
                        help="Choose what data to display from. training or testing data. ")
    parser.add_argument("--dims", type=int, default=2, choices=[2, 3],
                        help="View images analyzed as 2d independent slices or 3d volumetric images.")
    parser.add_argument("--single", type=bool, default=False, help="View one or 10 images at a time.")
    args = parser.parse_args()
    if args.dims == 3:
        V = VolumeViewer(f"datasets/preprocessed_quarter_size/{args.mode}/")
        V.view_volume(args.idx, args.focus)
    else:
        I = ImageViewer(f"datasets/preprocessed_2d/{args.mode}/")
        I.view_img(args.idx, focus=args.focus, singleview=args.single)
