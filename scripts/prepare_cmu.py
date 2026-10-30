import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import path
from src.data.dataloader_cmu import CMUDataset, _feature_cache_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Pack CMU-MOSI's 6.6k or CMU-MOSEI's 68.6k loose .npy feature files "
                    'into the single pkl the loader reads, so training never has to scan '
                    'them. Run once; the result is the only file for that dataset needed '
                    'afterwards.')
    parser.add_argument('--dataset', type=str, default='CMUMOSI', choices=['CMUMOSI', 'CMUMOSEI'])
    parser.add_argument('--audio-feature', type=str, default='wav2vec-large-c-UTT')
    parser.add_argument('--text-feature', type=str, default='deberta-large-4-UTT')
    parser.add_argument('--video-feature', type=str, default='manet_UTT')
    parser.add_argument('--force', action='store_true', help='Rebuild even if the file already exists.')
    args = parser.parse_args()

    feature_root = path.PATH_TO_FEATURES_Win[args.dataset]
    label_path = path.PATH_TO_LABEL_Win[args.dataset]
    roots = {m: os.path.join(feature_root, getattr(args, f'{m}_feature'))
             for m in ('audio', 'text', 'video')}

    out_path = _feature_cache_path(label_path, roots['audio'], roots['text'], roots['video'])

    if os.path.exists(out_path) and not args.force:
        size = os.path.getsize(out_path) / 1e6
        print(f'Already built: {out_path} ({size:.0f} MB)')
        print('Nothing to do -- pass --force to rebuild.')
        sys.exit(0)

    if args.force and os.path.exists(out_path):
        os.remove(out_path)

    print(f'Scanning feature files under {feature_root} ...')
    start = time.time()
    dataset = CMUDataset(label_path=label_path, audio_root=roots['audio'],
                         text_root=roots['text'], video_root=roots['video'])
    size = os.path.getsize(out_path) / 1e6

    print(f'\nWrote {out_path} ({size:.0f} MB) in {time.time() - start:.0f}s')
    print(f'  {len(dataset)} videos ({len(dataset.trainVids)} train / '
          f'{len(dataset.valVids)} valid / {len(dataset.testVids)} test), '
          f'dims audio/text/video = {dataset.adim}/{dataset.tdim}/{dataset.vdim}')
    print('\nThis file is self-contained: training reads it alone, so it is the only '
          'one you need to copy to another machine. Keep its name and directory, and '
          'pass the same --audio/--text/--video-feature values it was built with.')
