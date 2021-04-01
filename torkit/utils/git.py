from torch.utils.collect_env import run, run_and_read_all


def is_available():
    try:
        run('git version')
        return True
    except Exception as err:
        print(err)
        return False


def is_inside_work_tree():
    try:
        ret = run_and_read_all(run, 'git rev-parse --is-inside-work-tree')
        return ret == 'true'
    except Exception as err:
        print(err)
        return False


def get_head_hash(root_dir, first=8):
    git_rev = run_and_read_all(run, 'cd {:s} && git rev-parse HEAD'.format(root_dir))
    return git_rev[:first] if git_rev else git_rev


def get_modified_filenames(root_dir, git_dir):
    # Note that paths returned by git ls-files are relative to the script.
    return run_and_read_all(run, 'cd {:s} && git ls-files {:s} -m'.format(root_dir, git_dir))


def get_untracked_filenames(root_dir, git_dir):
    # Note that paths returned by git ls-files are relative to the script.
    return run_and_read_all(run, 'cd {:s} && git ls-files {:s} --exclude-standard --others'.format(root_dir, git_dir))


def collect_git_info(git_dir):
    if is_available():
        info_str = 'Git revision number: {}'.format(get_head_hash(git_dir))
        info_str += '\nGit modified files:\n{}'.format(get_modified_filenames(git_dir, git_dir))
        info_str += '\nGit untrakced files:\n{}'.format(get_untracked_filenames(git_dir, git_dir))
    else:
        info_str = 'Git is not available.'
    return info_str
