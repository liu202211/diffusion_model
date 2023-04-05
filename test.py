from collections import Counter


class Solution:
    def __init__(self):
        ...

    def balancedString(self, s: str) -> int:

        cnt, part = Counter(s), len(s) // 4
        if part == 0: return 0

        def check(cnt2):
            # 左包含 右不包含
            for key in ['Q', 'W', 'E', 'R']:
                if cnt.get(key, 0) - cnt2.get(key, 0) > part:
                    return False

            return True

        left = 0
        ret = len(s) + 1
        cnt2 = {'Q': 0, 'W':0, 'E':0, 'R': 0}
        for right in range(1, len(s) + 1):
            cnt2[s[right-1]] += 1
            if check(cnt2):
                while left < right and check(cnt2):
                    cnt2[s[left]] -= 1
                    left += 1
                ret = min(ret, right - left+1)

        return ret





if __name__ == '__main__':
    s = 'Q'*(4*10000)
    ret = Solution().balancedString(s)
    print(ret)