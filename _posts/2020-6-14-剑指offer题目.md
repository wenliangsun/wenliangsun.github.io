# 剑指offer题目及代码

## 数组中重复的数字(简单)

在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。[数组中重复的数字](https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/)

### 参考代码

```c++
int findRepeatNumber(vector<int>& nums) {
    unordered_map<int, int> hash;
    for (auto n : nums) {
        hash[n]++;
        if (hash[n] > 1) return n;
    }
    return 0;
}
```

## 二维数组中的查找(简单)

在一个 $n\times m$ 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。[二维数组中的查找](https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)

### 参考代码

```c++
bool findNumberIn2DArray(vector<vector<int>>& m, int t) {
    if (!m.size()) return false;
    int r = 0, c = m[0].size() - 1;
    while (r < m.size() && c >= 0) {
        if (m[r][c] == t)
            return true;
        else if (m[r][c] < t)
            r++;
        else
            c--;
    }
    return false;
}
```

## 替换空格(简单)

请实现一个函数，把字符串 s 中的每个空格替换成"%20"。[替换空格](https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/)

### 参考代码

```c++
string replaceSpace(string s) {
    string res;
    for (auto c : s) {
        if (c == ' ')
            res += "%20";
        else
            res += c;
    }
    return res;
}
```

## 从尾到头打印链表(简单)

输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。[从尾到头打印链表](https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)

### 参考代码

```c++
vector<int> res;
vector<int> reversePrint(ListNode* head) {
    dfs(head);
    return res;
}

void dfs(ListNode* head) {
    if (!head) return;
    dfs(head->next);
    res.push_back(head->val);
}
```

## 重建二叉树(中等)

输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。[重建二叉树](https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof/)

### 参考代码

```c++
unordered_map<int, int> pos;
TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
    int n = inorder.size();
    for (int i = 0; i < n; i++) pos[inorder[i]] = i;
    return dfs(preorder, inorder, 0, n - 1, 0, n - 1);
}

TreeNode* dfs(vector<int>& preorder, vector<int>& inorder, int pl, int pr,
                int il, int ir) {
    if (pl > pr) return nullptr;
    int val = preorder[pl];
    int k = pos[val];
    int len = k - il;
    auto root = new TreeNode(val);
    root->left = dfs(preorder, inorder, pl + 1, pl + len, il, k - 1);
    root->right = dfs(preorder, inorder, pl + len + 1, pr, k + 1, ir);
    return root;
}
```

## 用两个栈实现队列(简单)

用两个栈实现一个队列。[用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

### 参考代码

```c++
stack<int> s1;
stack<int> s2;
void appendTail(int value) { s1.push(value); }

int deleteHead() {
    if (s2.empty()) {
        while (s1.size()) {
            s2.push(s1.top());
            s1.pop();
        }
    }
    if (s2.size()) {
        int t = s2.top();
        s2.pop();
        return t;
    }
    return -1;
}
```

## 斐波拉契数列(简单)

写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项。[斐波拉契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/) 类似的还有青蛙跳台阶。

### 参考代码

```c++
int fib(int n) {
    if (n == 0) return 0;
    if (n == 1) return 1;
    int a = 0, b = 1;
    int res = 0;
    for (int i = 2; i <= n; i++) {
        res = a + b;
        if (res > 1e9 + 7) res -= (1e9 + 7);
        a = b;
        b = res;
    }
    return res > 1e9 + 7 ? res - (1e9 + 7) : res;
}
```

## 旋转数组的最小数字(简单)

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。[旋转数组的最小数字](https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)

### 参考代码

```c++
int minArray(vector<int>& numbers) {
    int l = 0, r = numbers.size() - 1;
    while (l < r) {
        int mid = l + r >> 1;
        if (numbers[mid] < numbers[r])
            r = mid;
        else if (numbers[mid] > numbers[r])
            l = mid + 1;
        else
            r--;
    }
    return numbers[l];
}
```

## 矩阵中的路径(中等)

请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。[矩阵中的路径](https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/)

### 参考代码

```c++
int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, -1, 0, 1};
int n, m;
bool exist(vector<vector<char>>& board, string word) {
    if (!board.size()) return false;
    n = board.size(), m = board[0].size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (board[i][j] == word[0] && dfs(board, i, j, 0, word))
                return true;
        }
    }
    return false;
}

bool dfs(vector<vector<char>>& board, int x, int y, int idx, string& word) {
    if (board[x][y] != word[idx]) return false;
    if (idx == word.size() - 1) return true;
    board[x][y] = '.';
    for (int i = 0; i < 4; i++) {
        int a = x + dx[i], b = y + dy[i];
        if (a >= 0 && a < n && b >= 0 && b < m)
            if (dfs(board, a, b, idx + 1, word)) return true;
    }
    board[x][y] = word[idx];
    return false;
}
```

## 机器人的运动范围(中等)

地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？[机器人的运动范围](https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/)

### 参考代码

```c++
int res = 0;
int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, -1, 0, 1};
vector<vector<bool>> st;
int movingCount(int m, int n, int k) {
    st = vector<vector<bool>>(m, vector<bool>(n));
    dfs(m, n, 0, 0, k);
    return res;
}

void dfs(int m, int n, int x, int y, int k) {
    if (digits(x) + digits(y) > k || st[x][y]) return;
    res++;
    st[x][y] = true;
    for (int i = 0; i < 4; i++) {
        int a = x + dx[i], b = y + dy[i];
        if (a >= 0 && a < m && b >= 0 && b < n) dfs(m, n, a, b, k);
    }
}

int digits(int n) {
    int res = 0;
    while (n) {
        res += n % 10;
        n /= 10;
    }
    return res;
}
```

## 剪绳子I(中等)

给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 `k[0],k[1]...k[m-1]` 。请问 `k[0]*k[1]*...*k[m-1]` 可能的最大乘积是多少？[剪绳子I](https://leetcode-cn.com/problems/jian-sheng-zi-lcof/)

### 参考代码

```c++
int cuttingRope(int n) { // 暴搜
    if (n == 2) return 1;
    int res = 0;
    for (int i = 1; i <= n; i++) {
        res = max(res, max(i * (n - i), i * cuttingRope(n - i)));
    }
    return res;
}

/**
 * 备忘录
 */
int cuttingRope(int n) {
    vector<int> memo(n + 1, -1);
    return dfs(n, memo);
}

int dfs(int n, vector<int>& memo) {
    if (n == 2) return 1;
    if (memo[n] != -1) return memo[n];
    int res = 0;
    for (int i = 1; i <= n; i++) {
        res = max(res, max(i * (n - i), i * dfs(n - i, memo)));
    }
    memo[n] = res;
    return res;
}

/**
 * 动态规划
 * 状态表示：dp[i]表示长度为i的绳子的能构成的最大乘积
 * 状态计算：dp[i] = max(i*(i-j),i*dp[i-j])
 * 1<=j<i每次都有两种选择，剩下一段绳子剪或者不剪
 */
int cuttingRope(int n) {
    vector<int> dp(n + 1);
    dp[2] = 1;
    for (int i = 3; i <= n; i++) {
        for (int j = 1; j < i; j++) {
            dp[i] = max(dp[i], max(j * (i - j), j * dp[i - j]));
        }
    }
    return dp[n];
}
```

## 剪绳子II(中等)

给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 `k[0],k[1]...k[m]` 。请问 `k[0]*k[1]*...*k[m]` 可能的最大乘积是多少？答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。[剪绳子II](https://leetcode-cn.com/problems/jian-sheng-zi-ii-lcof/)，[数学推导题解](https://leetcode-cn.com/problems/jian-sheng-zi-ii-lcof/solution/mian-shi-ti-14-ii-jian-sheng-zi-iitan-xin-er-fen-f/)

### 参考代码

```c++
int cuttingRope(int n) { //数论：数学推导 题解
    if (n <= 3) return n - 1;
    int a = n / 3, b = n % 3, p = 1e9 + 7;
    long long res = qmi(3, a - 1, p);
    if (b == 0) return res * 3 % p;
    if (b == 1) return res * 4 % p;
    return res * 6 % p;
}

/**
 * 贪心的思路
 */
int cuttingRope(int n) {
    if (n == 2) return 1;
    if (n == 3) return 2;
    int p = 1e9 + 7;
    long long res = 1;
    while (n > 4) {
        res *= 3;
        res %= p;
        n -= 3;
    }
    return res * n % p;
}

/**
 * 快速幂模板  求 m^k % p
 */
long long qmi(int m, int k, int p) {
    long long res = 1 % p, t = m;
    while (k) {
        if (k & 1) res = res * t % p;
        t = t * t % p;
        k >>= 1;
    }
    return res;
}
```

## 二进制中1的个数(简单)

请实现一个函数，输入一个整数，输出该数二进制表示中 1 的个数。[二进制中1的个数](https://leetcode-cn.com/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/)

### 参考代码

```c++
int hammingWeight(uint32_t n){
    int res = 0;
    while(n){
        res++;
        n = n & (n - 1); // 核心就这一句，每次消除n的最后一个1
    }
    return res;
}
```

## 数值的整数次方(中等)

实现函数double Power(double base, int exponent)，求base的exponent次方。不得使用库函数，同时不需要考虑大数问题。[数值的整数次方](https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/)

### 参考代码

```c++
double myPow(double x, int n){
    if (n == 0) return 1;
    if (x == 1) return x;
    double res = 1, t = x;
    double tn = n;
    long long absn = abs(n);
    // 快速幂
    while(absn){
        if (absn & 1) res = res * t;
        t = t * t;
        absn >>= 1;
    }
    return n > 0 ? res : 1 / res;
}
```

## 打印从1到最大的n位数(简单)

输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。[打印从1到最大的n位数](https://leetcode-cn.com/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof/)

### 参考代码

```c++
vector<int> printNumbers(int n){
    int mx = pow(10, n);
    vector<int> res;
    for (int i = 1; i < mx; i++) res.push_back(i);
    return res;
}
```

## 删除链表的节点(简单)

给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。[删除链表的节点](https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/)

### 参考代码

```c++
ListNode* deleteNode(ListNode* head, int val) {
    auto dummy = new ListNode(-1);
    dummy->next = head;
    auto p = dummy;
    auto cur = head;
    while(cur) {
        if(cur->val == val){
            p->next = cur->next;
            cur = cur->next;
        } else{
            p = p->next;
            cur = cur->next;
        }
    }
    return dummy->next;
}
```

## 正则表达式匹配(困难)

请实现一个函数用来匹配包含'. '和'\*'的正则表达式。模式中的字符'.'表示任意一个字符，而'\*'表示它前面的字符可以出现任意次（含0次）。在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*\ac\*a"匹配，但与"aa.a"和"ab\*a"均不匹配。[正则表达式匹配](https://leetcode-cn.com/problems/zheng-ze-biao-da-shi-pi-pei-lcof/)

### 参考代码

```c++
class Solution {
   public:
    /**
     * 动态规划
     * 状态表示：dp[i][j]表示s的前i个字符和p的前j个字符是否匹配
     * 状态计算：分两部分，如果当前字符匹配，则匹配下一个
     *  如果p中当前字符是*，则判断*前面的字符与s中的当前字符是否相等，如果相等，则可以匹配多次1次或0次，如果不相等则匹配0次
     */
    bool isMatch(string s, string p) {
        int n = s.size(), m = p.size();
        vector<vector<bool>> dp(n + 1, vector<bool>(m + 1));
        dp[0][0] = true;  // 空字符串匹配
        // 空字符串和a*a*a*a*a*...匹配的情况
        for (int i = 2; i <= m; i += 2)
            if (p[i - 1] == '*') dp[0][i] = dp[0][i - 2];

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                // 当前字符匹配
                if (s[i - 1] == p[j - 1] || p[j - 1] == '.')
                    dp[i][j] = dp[i - 1][j - 1];
                // 下一个字符是*
                if (p[j - 1] == '*') {
                    // *前面的字符和s中的当前字符匹配
                    if (j > 1 && (p[j - 2] == s[i - 1] || p[j - 2] == '.'))
                        // 匹配多次1次0次
                        dp[i][j] = dp[i - 1][j] || dp[i][j - 1] || dp[i][j - 2];
                    else
                        dp[i][j] = dp[i][j - 2];
                }
            }
        }
        return dp[n][m];
    }

    /**
     * 回溯法
     */
    bool isMatch(string s, string p) { return dfs(s, 0, p, 0); }
    /**
     * 分两种情况：
     * 1.下一个字符是*，如果当前字符匹配（或当前模式串是'.'），则可以匹配0次1次多次，如果当前字符不匹配，则匹配0次
     * 2.下一个字符不是*，如果当前字符匹配，则匹配下一个字符，如果不匹配，返回false
     */
    bool dfs(string& s, int sidx, string& p, int pidx) {
        // 匹配完成
        if (sidx == s.size() && pidx == p.size()) return true;
        // 模式串没有了，匹配失败
        if (sidx < s.size() && pidx == p.size()) return false;
        // 下一个字符是*
        if (pidx + 1 < p.size() && p[pidx + 1] == '*')
            // 当前字符匹配
            if (sidx < s.size() && (s[sidx] == p[pidx] || p[pidx] == '.'))
                // 匹配多次，1次，0次
                return dfs(s, sidx + 1, p, pidx) ||
                       dfs(s, sidx + 1, p, pidx + 2) ||
                       dfs(s, sidx, p, pidx + 2);
            else
                // 当前字符不匹配，则匹配0次
                return dfs(s, sidx, p, pidx + 2);
        // 下一个字符不是*，判断当前字符是否匹配
        if (sidx < s.size() && (s[sidx] == p[pidx] || p[pidx] == '.'))
            return dfs(s, sidx + 1, p, pidx + 1);
        return false;
    }
};
```

## 表示数值的字符串(中等)

请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。[表示数值的字符串](https://leetcode-cn.com/problems/biao-shi-shu-zhi-de-zi-fu-chuan-lcof/)

### 参考代码

```c++
bool isNumber(string s) {
    if (s.empty()) return false;
    int idx = 0;
    while (s[idx] == ' ') idx++;
    if (s[idx] == '+' || s[idx] == '-') idx++;
    int n_dot, n_num;
    for (n_dot = 0, n_num = 0; (s[idx] >= '0' && s[idx] <= '9') || s[idx] == '.'; idx++)
        s[idx] == '.' ? n_dot++ : n_num++;
    if (n_dot > 1 || n_num < 1) return false;
    if (s[idx] == 'e' || s[idx] == 'E') {
        idx++;
        if (s[idx] == '+' || s[idx] == '-') idx++;
        n_num = 0;
        for (; s[idx] >= '0' && s[idx] <= '9'; idx++) n_num++;
        if (n_num < 1) return false;
    }
    while (s[idx] == ' ') idx++;
    return idx == s.size();
}
```

## 调整数组顺序使奇数位于偶数前面(简单)

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。[调整数组顺序使奇数位于偶数前面](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

### 参考代码

```c++
vector<int> exchange(vector<int>& nums) {
    vector<int> res; // 不改变原数组
    for (auto n : nums) if (n & 1) res.push_back(n);
    for (auto n : nums) if(!(n & 1)) res.push_back(n);
    return res;
}

/**
 * 双指针，改变原数组
 */
vector<int> exchange(vector<int>& nums) {
    int l = 0, r = nums.size() - 1;
    while (l < r) {
        if (nums[l] & 1) {
            l++;
            continue;
        }
        if (!(nums[r] & 1)){
            r--;
            continue;
        }
        swap(nums[l++], nums[r--]);
    }
    return nums;
}
```

## 链表中倒数第k个结点(简单)

输入一个链表，输出该链表中倒数第k个节点。[链表中倒数第k个结点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

### 参考代码

```c++
ListNode* getKthFromEnd(ListNode* head, int k) {
    ListNode*s, *f;
    f = s = head;
    for (int i = 0; i < k; i++) f = f->next;
    while (f) {
        s = s->next;
        f = f->next;
    }
    return s;
}
```

## 反转链表(简单)

定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。[反转链表](https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/)

### 参考代码

```c++
ListNode* reverseList(ListNode* head) { // 迭代版本
    ListNode* pre = nullptr;
    ListNode* cur = head;
    while (cur) {
        auto nx = cur->next;
        cur->next = pre;
        pre = cur;
        cur = nx;
    }
    return pre;
}
/**
 * 递归版本
 */
ListNode* reverseList(ListNode* head) {
    if (!head || !head->next) return head;
    auto last = reverseList(head->next);
    head->next->next = head;
    head->next = nullptr;
    return last;
}
```

## 合并两个排序链表(简单)

输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。[合并两个排序链表](https://leetcode-cn.com/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/)

### 参考代码

```c++
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2){
    auto dummy = new ListNode(-1);
    auto p = dummy;
    while (l1 && l2) {
        if (l1->val < l2->val) {
            p->next = l1;
            l1 = l1->next;
        } else {
            p->next = l2;
            l2 = l2->next;
        }
        p = p->next;
    }
    while (l1) {
        p->next = l1;
        l1 = l1->next;
        p = p->next;
    }
    while (l2) {
        p->next = l2;
        l2 = l2->next;
        p = p->next;
    }
    return dummy->next;
}
```

## 树的子结构(中等)

输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)[树的子结构](https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/)

### 参考代码

```c++
bool isSubStructure(TreeNode* A, TreeNode* B) {
    return (A && B) && (dfs(A, B) || isSubStructure(A->left, B) || isSubStructure(A->right, B));
}

bool dfs(TreeNode* p1, TreeNode* p2) {
    if (!p2) return true;
    if (!p1) return false;
    if (p1->val != p2->val) return false;
    return dfs(p1->left, p2->left) && dfs(p1->right, p2->right);
}
```

## 二叉树的镜像(简单)

请完成一个函数，输入一个二叉树，该函数输出它的镜像。[二叉树的镜像](https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/)

### 参考代码

```c++
TreeNode* mirrorTree(TreeNode* root) {
    if (!root) return root;
    auto t = root->left;
    root->left = mirrorTree(root->right);
    root->right = mirrorTree(t);
    return root;
}
```

## 对称的二叉树(简单)

请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。[对称的二叉树](https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/)

### 参考代码

```c++
bool isSymmetric(TreeNode* root) {
    return dfs(root, root);
}

bool dfs(TreeNode* r1, TreeNode* r2) {
    if (!r1 && !r2) return true;
    if (!r1 || !r2) return false;
    if (r1->val != r2->val) return false;
    return dfs(r1->left, r2->right) && dfs(r1->right, r2->left); 
}
```

## 顺时针打印矩阵(简单)

输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。[顺时针打印矩阵](https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/)

### 参考代码

```c++
vector<int> spiralOrder(vector<vector<int>>& matrix) {
    vector<int> res;
    if (!matrix.size()) return res;
    int n = matrix.size(), m = matrix[0].size();
    vector<vector<bool>> st(n, vector<bool>(m));
    int dx[4] = {0, 1, 0, -1}, dy[4] = {1, 0, -1, 0};
    int x = 0, y = 0, d = 0;
    for (int i = 1; i <= n * m; i++){
        int nx = x + dx[d], ny = y + dy[d];
        if (nx < 0 || nx >= n || ny < 0 || ny >= m || st[nx][ny]){
            d = (d + 1) % 4;
            nx = x + dx[d], ny = y + dy[d];
        }
        res.push_back(matrix[x][y]);
        st[x][y] = true;
        x = nx, y = ny;
    }
    return res;
}
```

## 包含min函数的栈(简单)

定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。[包含min函数的栈](https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/)

### 参考代码

```c++
stack<int> data;
stack<int> m;
MinStack() {

}

void push(int x) {
    data.push(x);
    if (m.empty() || x < m.top())
        m.push(x);
    else
        m.push(m.top());
}

void pop() {
    auto t = data.top();
    data.pop();
    m.pop();
}

int top() {
    return data.top();
}

int min() {
    return m.top();
}
```

## 栈的压入弹出序列(中等)

输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。[栈的压入弹出序列](https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/)

### 参考代码

```c++
bool validateStackSequences(vector<int>& pushed, vector<int>& popped){
    stack<int> stk;
    int idx = 0;
    for (int i = 0; i < pushed.size(); i++){
        stk.push(pushed[i]);
        while (!stk.empty() && stk.top() == popped[idx]) {
            stk.pop();
            idx++;
        }
    }
    return stk.empty();
}
```

## 从上到下打印二叉树I/II/III(简单)

从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。[从上到下打印二叉树I](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/)

### 参考代码

```c++
vector<int> levelOrder(TreeNode* root) {
    vector<int> res;
    if (!root) return res;
    queue<TreeNode*> q;
    q.push(root);
    while(q.size()){
        int len = q.size();
        for (int i = 0; i < len; i++ ){
            auto t = q.front();
            q.pop();
            res.push(t->val);
            if (t->left) q.push(t->left);
            if (t->right) q.push(t->right);
        }
    }
    return res;
}

vector<vector<int>> levelOrder(TreeNode* root) {
    vector<vector<int>> res;
    if (!root) return res;
    queue<TreeNode*> q;
    q.push(root);
    while (q.size()) {
        int len = q.size();
        vector<int> t;
        for (int i = 0; i < len; i++) {
            auto node = q.front();
            q.pop();
            t.push_back(node->val);
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        res.push_back(t);
    }
    return res;
}

vector<vector<int>> levelOrder(TreeNode* root) {
    vector<vector<int>> res;
    if (!root) return res;
    queue<TreeNode*> q;
    q.push(root);
    bool d = false;
    while (q.size()) {
        vector<int> t;
        int len = q.size();
        for (int i = 0; i < len; i++) {
            auto node = q.front();
            q.pop();
            t.push_back(node->val);
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        if (d) reverse(t.begin(), t.end());
        res.push_back(t);
        d = !d;
    }
    return res;
}
```

## 二叉搜索树的后序遍历序列(中等)

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 true，否则返回 false。假设输入的数组的任意两个数字都互不相同。[二叉搜索树的后序遍历序列](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)

### 参考代码

```c++
bool verifyPostorder(vector<int>& postorder) {
    return dfs(postorder, 0, postorder.size() - 1);
}

/**
 * 左区间都小于最后一个数，右区间都大于最后一个数
 */
bool dfs(vector<int>& preorder, int l, int r) {
    if (l >= r) return true;
    int p = l;
    while (postorder[p] < postorder[r]) p++; // 统计左区间
    int m = p;
    while (postorder[p] > postorder[r]) p++; // 统计右区间
    return (p == r) && dfs(postorder, l, m - 1) && dfs(postorder, m + 1, r);
}
```

## 二叉树中和为某一值的路径(中等)

输入一棵二叉树和一个整数，打印出二叉树中节点值的和为输入整数的所有路径。从树的根节点开始往下一直到叶节点所经过的节点形成一条路径。[二叉树中和为某一值的路径](https://leetcode-cn.com/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/)

### 参考代码

```c++
vector<vector<int>> res;
vector<vector<int>> pathSum(TreeNode* root, int sum) {
    if (!root) return res;
    vector<int> path;
    dfs(root, sum, path);
    return res;
}

void dfs(TreeNode* root, int sum, vector<int>& path) {
    if (!root) return;
    path.push_back(root->val);
    if (!root->left && !root->right && sum - root->val == 0) res.push_back(path);
    dfs(root->left, sum - root->val, path);
    dfs(root->right, sum - root->val, path);
    path.pop_back();
}
```

## 复杂链表的复制(中等)

请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。[复杂链表的复制](https://leetcode-cn.com/problems/fu-za-lian-biao-de-fu-zhi-lcof/)

### 参考代码

```c++
Node* copyRandomList(Node* head) {
    // 复制结点
    for (auto p = head; p;) {
        auto np = new Node(p->val);
        auto nx = p->next;
        p->next = np;
        np->next = nx;
        p = nx;
    }
    // 更新Random指针
    for (auto p = head; p; p = p->next->next) {
        if (p->random) p->next->random = p->random->next;
    }
    // 拆分链表
    auto dummy = new Node(-1);
    auto cur = dummy;
    for (auto p = head; p;) {
        auto pre = p;  // 用于复原原来的链表
        cur->next = p->next;
        cur = cur->next;
        p = p->next->next;
        pre->next = p;
    }
    return dummy->next;
}
```

## 二叉搜索树与双向链表(中等)

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。[二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

思路：利用二叉搜索树中序遍历的性质，对于当前结点，我们先中序遍历转换其左子树，此时得到的结果就是左子树部分转换的双向链表，记末尾结点为pre，则对于当前结点cur，令cur的左指针指向pre，pre的右指针指向cur，更新pre为当前结点cur，然后遍历当前结点的右子树。当pre为null是，此时cur为头结点。

### 参考代码

```c++
Node* pre = nullptr; // 记录前一个结点
Node* head; // 记录头结点
Node* treeToDoublyList(Node* root){
    if (!root) return root;
    // 下面这两行转换成循环双向链表
    head->left = pre;
    pre->right = head;
    return head;
}

void dfs(Node* cur) {
    if (!cur) return;
    dfs(cur->left); // 先遍历左子树
    if (pre) pre->right = cur; // 如果pre不为空，则连接当前结点
    else head = cur; //否则，当前结点是头结点
    cur->left = pre;
    pre = cur; // 更新前一个结点为当前结点
    dfs(cur->right); // 遍历右子树
}
```

## 序列化二叉树(困难)

请实现两个函数，分别用来序列化和反序列化二叉树。[序列化二叉树](https://leetcode-cn.com/problems/xu-lie-hua-er-cha-shu-lcof/)

### 参考代码

```c++
string serialize(TreeNode* root) {
    string res;
    dfs1(root, res);
    return res;
}

void dfs(TreeNode* root, string& res){
    if(!root) {
        res += "#,";
        return;
    }
    res += to_string(root->val) + ',';
    dfs1(root->left, res);
    dfs1(root->right, res);
}

TreeNode* deserialize(string data) {
    int idx = 0;
    return dfs2(data, idx);
}

TreeNode* dfs2(string& data, int& idx) {
    if (data[idx] == '#') {
        idx += 2;
        return nullptr;
    }
    int t = 0;
    bool is_minus = false;
    if (data[idx] == '-') {
        idx++;
        is_minus = true;
    }
    while (data[idx] != ',') {
        t = t * 10 + data[idx] - '0';
        idx++;
    }
    idx++;
    if (is_minus) t = -t;
    auto root = new TreeNode(t);
    root->left = dfs2(data, idx);
    root->right = dfs2(data, idx);
    return root;
}
```

## 字符串的排列(中等)

输入一个字符串，打印出该字符串中字符的所有排列。你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。[字符串的排列](https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/)

### 参考代码

```c++
vector<string> res;
vector<bool> st;
vector<string> permutation(string s) {
    string path;
    sort(s.begin(), s.end()); // 去重需要排序，让相同的元素在一起
    st = vector<bool>(s.size());
    dfs(s, 0, path);
    return res;
}

void dfs(string& s, int idx, string& path) {
    if (idx == s.size()) {
        res.push_back(path);
        return;
    }
    for (int i = 0; i < s.size(); i++) {
        // 注意这儿最后加！的话剪枝更彻底，不加！也可以
        if (st[i] || (i > 0 && s[i] == s[i - 1] && !st[i - 1])) continue;
        st[i] = true;
        path.push_back(s[i]);
        dfs(s, idx + 1, path);
        st[i] = false;
        path.pop_back();
    }
}
```

## 数组中出现次数超过一半的数字(简单)


数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。你可以假设数组是非空的，并且给定的数组总是存在多数元素。[数组中出现次数超过一半的数字](https://leetcode-cn.com/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/)

### 参考代码

```c++
int majorityElement1(vector<int>& nums) {
    sort(nums.begin(), nums.end());
    return nums[nums.size()/2];
}

/**
 * 摩尔投票
 */
int majorityElement(vector<int>& nums) {
    int cnt = 0, val = nums[0];
    // 抵消阶段
    for (auto n : nums) {
        if (!cnt) val = n, cnt = 1;
        else {
            if (n == val) cnt++;
            else cnt--;
        }
    }
    cnt = 0;
    // 验证阶段
    for (auto n : nums) if (n == val) cnt++;
    return cnt > nums.size() / 2 ? val : 0;
}

/**
 * 哈希表
 */
int majorityElement2(vector<int>& nums) {
    unordered_map<int, int> hash;
    for (auto n : nums) hash[n]++;
    int res = 0;
    for (auto item : hash) {
        if (item.second > (nums.size() / 2)) {
            res = item.first;
        }
    }
    return res;
}
```

## 最小的k个数(简单)

输入整数数组 arr ，找出其中最小的 k 个数。[最小的k个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)

### 参考代码

```c++
vector<int> getLeastNumbers(vector<int>& arr, int k) {
    priority_queue<int, vector<int>, less<int>> q;
    vector<int> res;
    for(auto n : arr) {
        // 维护一个长度为k的大顶堆
        if (q.size() < k) q.push(n);
        else {
            if(q.size() && n < q.top()) q.pop(), q.push(n);
        }
    }
    while(q.size()) {
        res.push_back(q.top());
        q.pop();
    }
    return res;
}
```

## 数据流中的中位数(困难)

如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。[数据流中的中位数](https://leetcode-cn.com/problems/shu-ju-liu-zhong-de-zhong-wei-shu-lcof/)

### 参考代码

```c++
priority_queue<int, vector<int>, greater<int>> minq; // 最小堆
priority_queue<int> maxq; // 最大堆
int cnt = 0;
MedianFinder() {
    cnt = 0;
}

void addNum(int num) {
    cnt++;
    maxq.push(num);
    minq.push(maxq.top());
    maxq.pop();
    if (cnt & 1) {
        maxq.push(minq.top());
        minq.pop();
    }
}

double findMedian() {
    if (cnt & 1) return maxq.top();
    else return ((double)maxq.top() + minq.top()) / 2;
}
```

## 连续子数组的最大和(简单)

输入一个整型数组，数组里有正数也有负数。数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。[连续子数组的最大和](https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)

### 参考代码

```c++
int maxSubArray(vector<int>& nums) {
    // 按照动态规划的思路走
    int last = 0, res = INT_MIN;
    for (auto n : nums) {
        int t = max(last, 0) + n;
        res = max(res, t);
        last = t;
    }
    return res;
}
```

## 1~n整数中1出现的次数(中等)

输入一个整数 n ，求1～n这n个整数的十进制表示中1出现的次数。[1~n整数中1出现的次数](https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/)

### 参考代码

```c++
待填。。。
```

## 数字序列中某一位的数字(中等)

数字以0123456789101112131415…的格式序列化到一个字符序列中。在这个序列中，第5位（从下标0开始计数）是5，第13位是1，第19位是4，等等。[数字序列中某一位的数字](https://leetcode-cn.com/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/)

### 参考代码

```c++
int findNthDigit(int n) {
    if (n < 10) return n;
    int base = 1;
    long cnt = 0;
    while (true) {
        cnt = helper(base);
        if (n < cnt) break;
        n -= cnt;
        base++;
    }
    int num = (int) (n / base + pow(10, base - 1)); // 找到具体的数字
    string t = to_string(num); // 将数字转成字符串
    return t[n % base] - '0';
}
// 计算当前base有多少位
long helper(int base) {
    if (base == 1) return 10;
    return 9ll * pow(10, base - 1) * base;
}
```

## 把数组排成最小的数(中等)

输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。[把数组排成最小的数](https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/)

### 参考代码

```c++
static bool cmp(int a, int b) {
    string s1 = to_string(a) + to_string(b);
    string s2 = to_string(b) + to_string(a);
    return s1 < s2;
}

string minNumber(vector<int>& nums) {
    // 关键在于排序规则的定义
    sort(nums.begin(), nums.end(), cmp);
    string res;
    for (auto n : nums) res += to_string(n);
    return res;
}
```

## 把数字翻译成字符串(中等)

给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。[把数字翻译成字符串](https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/)

### 参考代码

```c++
int res = 0;
int translateNum(int num) {
    string s = to_string(num);
    dfs(s, 0);
    return res;
}

void dfs(string& s, int idx) {
    if (idx == s.size()) {
        res++;
        return;
    }
    for (int i = idx; i < s.size(); i++) {
        // 注意这儿的剪枝
        string t = s.substr(idx, i - idx + 1);
        if (idx < i && t[0] == '0') continue; // 数字的第一个位置不能为0
        int n = stoi(t);
        if (n > 25) break; // 当前数字不能大于25
        dfs(s, i + 1);
    }
}
```

## 礼物的最大价值(中等)

在一个 $m\times n$ 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物。[礼物的最大价值](https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof/)

### 参考代码

```c++
int maxValue(vector<vector<int>>& grid) {
    int n = gird.size(), m = gird[0].size();
    // dp[i][j] 表示到达位置(i,j)的所有方案，其值表示方案的最大价值
    vector<vector<int>> dp(n + 1, vector<int>(m + 1));
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            // 状态转移，可以从右侧，和上侧过来
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]) + grid[i - 1][j - 1];
        }
    }
    return dp[n][m];
}
```

## 最长不含重复字符的子字符串(中等)

请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。[最长不含重复字符的子字符串](https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/)

### 参考代码

```c++
int lengthOfLongestSubstring(string s) {
    unordered_map<char, int> hash;
    int res = 0;
    // 双指针或滑动窗口
    for (int i = 0, j = 0; i < s.size(); i++) {
        hash[s[i]]++;
        while (hash[s[i]] > 1) hash[s[j++]]--;
        res = max(res, i - j + 1);
    }
    return res;
}
```

## 丑数(中等)

我们把只包含因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。[丑数](https://leetcode-cn.com/problems/chou-shu-lcof/)

思路：不能直接依次生成数字，然后判断是否是丑数.我们依次生成从小到大的丑数,考虑到后面的丑数是前面的丑数乘以2,3,5得到的，可以对当前的丑数分别乘以2,3,5。然后取大于当前生成的最大丑数中的最小丑数。此时，考虑已生成的丑数在前面的某个位置之后的丑数乘以2获得的丑数会大于当前生成的丑数，我们记录这个位置。同样，3和5也有类似的位置，因此我们分别记录这三个位置，然后分别乘以2,3,5取其中的最小值,再更新这三个位置。

### 参考代码

```c++
int nthUglyNumber(int n) {
    if (n < 0) return 0;
    vector<int> tmp(n);
    tmp[0] = 1;
    int idx = 1;
    int idx2 = 0, idx3 = 0, idx5 = 0;
    while (idx < n) {
        tmp[idx] = min(tmp[idx2] * 2, min(tmp[idx3] * 3, tmp[idx5] * 5));
        while (tmp[idx2] * 2 <= tmp[idx]) idx2++;
        while (tmp[idx3] * 3 <= tmp[idx]) idx3++;
        while (tmp[idx5] * 5 <= tmp[idx]) idx5++;
        idx++;
    }
    return tmp[n-1];
}
```

## 第一个只出现一次的字符(简单)

在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。[第一个只出现一次的字符](https://leetcode-cn.com/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/)

### 参考代码

```c++
char firstUniqChar(string s) {
    int t[26] = {0};
    for (auto c : s) t[c - 'a']++;
    for (auto c : s)
        if (t[c - 'a'] == 1) return c;
    return ' ';
}
```

## 数组中的逆序对(困难)

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。[数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

### 参考代码

```c++
int res = 0;
int reversePairs(vector<int>& nums) {
    vector<int> t(nums.size());
    helper(nums, 0, nums.size() - 1, t);
    return res;
}
// 归并思路
// 首先将数组划分为单个，然后进行二路归并比较，统计逆序对，并对统计过的区间进行排序。
void helper(vector<int>& nums, int l, int r, vector<int>& t) {
    if (l >= r) return;
    int mid = l + r >> 1;
    helper(nums, l, mid, t);
    helper(nums, mid + 1, r, t);
    int idx = r, idxl = mid, idxr = r;
    while (idxl >= l && r >= mid + 1) {
        if (nums[idxl] > nums[r]) {
            res += r - mid;
            t[idx--] = nums[idxl--];
        } else {
            t[idx--] = nums[r--];
        }
    }
    while (idxl >= l) t[idx--] = nums[idxl--];
    while (r >= mid + 1) t[idx--] = nums[r--];
    while (idxr >= l) {
        nums[idxr] = t[idxr];
        idxr--;
    }
}
```

## 两个链表的第一个公共结点(简单)

输入两个链表，找出它们的第一个公共节点。[两个链表的第一个公共节点](https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/)

### 参考代码

```c++
ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
    auto p = headA, q = headB;
    while (p != q) {
        if (p) p = p->next;
        else p = headB;
        if (q) q = q->next;
        else q = headA;
    }
    return p;
}
```

## 在排序数组中查找数字I(简单)

统计一个数字在排序数组中出现的次数。[在排序数组中查找数字I](https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)

### 参考代码

```c++
int search(vector<int>& nums, int target) {
    if (!nums.size()) return 0;
    // 二分的两种模板
    int l = 0, r = nums.size() - 1;
    while (l < r) {
        int mid = l + r >> 1;
        if (nums[mid] >= target) r = mid;
        else l = mid + 1;
    }
    if (nums[l] != target) return 0;
    int left = l;
    l = 0, r = nums.size() - 1;
    while (l < r) {
        int mid = l + r + 1 >> 1;
        if (nums[mid] <= target) l = mid;
        else r = mid - 1;
    }
    return l - left + 1;
}
```

## 0~n-1中缺失的数字(简单)

一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。[0~n-1中缺失的数字](https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/)

### 参考代码

```c++
int missingNumber(vector<int>& nums) {
    int res = nums.size();
    for (int i = 0; i < nums.size(); i++) {
        res ^= i ^ nums[i];
    }
    return res;
}
```

## 二叉搜索树的第k大结点(简单)


给定一棵二叉搜索树，请找出其中第k大的节点。[二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

### 参考代码

```c++
int res = 0, cnt = 0;
int kthLargest(TreeNode* root, int k) {
    dfs(root, k);
    return res;
}

void dfs(TreeNode* root, int k) {
    if (!root) return;
    dfs(root->right, k); // 第k小就先遍历左子树
    cnt++;
    if (cnt == k) {
        res = root->val;
        return;
    }
    dfs(root->left, k);
}
```

## 二叉树的深度(简单)

输入一棵二叉树的根节点，求该树的深度。[二叉树的深度](https://leetcode-cn.com/problems/er-cha-shu-de-shen-du-lcof/)

### 参考代码

```c++
int maxDepth(TreeNode* root) {
    if (!root) return 0;
    int l = maxDepth(root->left);
    int r = maxDepth(root->right);
    return max(l, r) + 1;
}
```

## 平衡二叉树(简单)

输入一棵二叉树的根节点，判断该树是不是平衡二叉树。如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。[平衡二叉树](https://leetcode-cn.com/problems/ping-heng-er-cha-shu-lcof/)

### 参考代码

```c++
bool isBalanced(TreeNode* root) {
    if (!root) return true;
    int l = dfs(root->left);
    int r = dfs(root->right);
    return abs(l - r) <= 1 && isBalanced(root->left) && isBalanced(root->right);
}
// 求最大深度
int dfs(TreeNode* root) {
    if (!root) return 0;
    int l = dfs(root->left);
    int r = dfs(root->right);
    return max(l, r) + 1;
}
```

## 数组中数字出现的次数(中等)

一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。[数组中数字出现的次数](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/)

思路：只有一个只出现一次的数字，其他数字都出现了两次，则可通过异或求得该数字。而有两个只出现一次的数字，我们可以考虑将数组分成两部分，每一部分中都只有一个数字出现了一次，其余数字都出现了两次，然后分别用异或求数字：1.对所有数字进行异或，可以得到一个数字，该数字一定不为0，说明其中必定有一位是1，然后以这一位是否为1作为划分规则，划分数据（出现两次的数字一定会出现在同一组里，因为相同的数字他们的位是一样的）2.然后对两部分数据分别进行异或运算即可求得数字

### 参考代码

```c++
vector<int> singleNumbers(vector<int>& nums) {
    int n1 = 0, n2 = 0;
    int t = 0;
    for (auto n : nums) t ^= n;
    int idx = 0;
    while (!(t & 1)) {
        t >>= 1;
        idx++;
    }
    for (auto n : nums) {
        if ((n >> idx) & 1) n1 ^= n;
        else n2 ^= n;
    }
    return {n1, n2};
}
```

## 数组中数字出现的次数II(中等)

在一个数组 nums 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。[数组中数字出现的次数II](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-ii-lcof/)

思路：因为相同的数字出现了三次，因此不能使用异或操作来做。我们考虑如果一个数字出现3次，那么它的二进制的每一位也会出现3次，如果把所有出现3次的数字的二进制表示的每一位都分别加起来，那么每一位都能被3整除。因此，我们将数组中的所有数字的二进制表示的每一位都加起来，如果某一位的和能被3整除，那么说明只出现一次的数字在该位是0，否则是1

### 参考代码

```c++
int singleNumber(vector<int>& nums) {
    int m[32] = {0};
    for (auto n : nums) {
        for (int i = 0; i < 31; i++) {
            if ((n >> i) & 1) m[i]++;
        }
    }
    int res = 0;
    for (int i = 31; i >= 0; i--) {
        res = res << 1;
        res += m[i] % 3;
    }
    return res;
}
```

## 和为S的两个数字(简单)

输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。[ 和为s的两个数字](https://leetcode-cn.com/problems/he-wei-sde-liang-ge-shu-zi-lcof/)

### 参考代码

```c++
vector<int> twoSum(vector<int>& nums, int target) {
    int l = 0, r = nums.size() - 1;
    while (l < r) {
        int sum = nums[l] + nums[r];
        if (sum == target) return {nums[l], nums[r]};
        else if (sum < target) l++;
        else r--;
    }
    return {-1, -1};
}
```

## 和为s的连续正序列(中等)

输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。[和为s的连续正数序列](https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

### 参考代码

```c++
vector<vector<int>> findContinuousSequence(int target) {
    vector<vector<int>> res;
    int sum = 0;
    for (int i = 1; i <= target / 2; i++) {
        for (int j = i; ; j++) {
            sum += j;
            if (sum > target) {
                sum = 0;
                break;
            } else if (sum == target) {
                vector<int> t;
                for (int k = i; k <= j; k++) t.push_back(k);
                res.push_back(t);
                sum = 0;
                break;
            }
        }
    }
    return res;
}
```

## 翻转单词序列(简单)

输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，则输出"student. a am I"。[翻转单词顺序](https://leetcode-cn.com/problems/fan-zhuan-dan-ci-shun-xu-lcof/)

### 参考代码

```c++
string reverseWords(string s) {
    int k = 0; // 记录新复制的位置
    for (int i = 0; i < s.size(); i++) {
        while (i < s.size() && s[i] == ' ') i++; // 去掉前面的空格
        if (i == s.size()) break;
        int j = i;
        while (j < s.size() && s[j] != ' ') j++; // 寻找单词
        reverse(s.begin() + i, s.begin() + j); // 翻转单词
        if (k) s[k++] = ' '; // 添加空格
        while (i < j) s[k++] = s[i++]; // 拷贝到新的位置
    }
    s.erase(s.begin() + k, s.end()); // 删除多余的空格
    reverse(s.begin(), s.end()); //翻转整个字符串
    return s;
}
```

## 左旋转字符串(简单)

字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。[ 左旋转字符串](https://leetcode-cn.com/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/)

### 参考代码

```c++
string reverseLeftWords(string s, int n) {
    reverse(s.begin(), s.begin() + n);
    reverse(s.begin() + n, s.end());
    reverse(s.begin(), s.end());
    return s;
}
```

## 滑动窗口的最大值(困难)

给定一个数组 nums 和滑动窗口的大小 k，请找出所有滑动窗口里的最大值。[滑动窗口的最大值](https://leetcode-cn.com/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/)

### 参考代码

```c++
vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    deque<int> q; // 维护一个单调递减的队列
    vector<int> res;
    for (int i = 0; i < nums.size(); i++) {
        // 如果队头元素不在当前区间里面，则出队
        if (q.size() && i - k + 1 > q.front()) q.pop_front();
        // 如果队尾元素比当前元素小，则队尾元素出队
        while (q.size() && nums[q.back()] <= nums[i]) q.pop_back();
        q.push_back(i);
        // 当区间满足时，往答案数组中添加队头元素
        if (i >= k - 1) res.push_back(nums[q.front()]);
    }
    return res;
}
```

## 队列的最大值(中等)

请定义一个队列并实现函数 max_value 得到队列里的最大值，要求函数max_value、push_back 和 pop_front 的均摊时间复杂度都是O(1)。若队列为空，pop_front 和 max_value 需要返回 -1。[队列的最大值](https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof/)

### 参考代码

```c++
deque<int> dq;
queue<int> q;

int max_value() {
    if (dq.size())
        return dq.front();
    else
        return -1;
}

/**
 * 维护一个单调的双端队列
 */
void push_back(int value) {
    while (dq.size() && dq.back() < value) dq.pop_back();
    dq.push_back(value);
    q.push(value);
}

int pop_front() {
    if (q.size()) {
        auto t = q.front();
        q.pop();
        if (t == dq.front()) dq.pop_front();
        return t;
    } else
        return -1;
}
```

## n个骰子的点数(简单)

把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。[n个骰子的点数](https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/)

### 参考代码

```c++
vector<double> twoSum(int n) {
    /**
     * 动态规划：
     * 状态表示：dp[i][j] 表示第i次掷之后出现点数j的所有情况的集合，其值表示点数是j的数量
     * 状态计算：dp[i][j]可以由状态dp[i-1][j],dp[i-1][j-2],dp[i-1][j-3]...dp[i-1][j-6]计算得到。
     */
    vector<vector<int>> dp(n + 1, vector<int>(6 * n + 1));
    for (int i = 1; i <= 6; i++) dp[1][i] = 1; // 初始化投掷一次的情况
    for (int i = 2; i <= n; i++) {  // 遍历投掷次数
        for (int j = i; j <= 6 * i; j++) {  // 遍历当前投递可能出现的点数
            for (int k = 1; k <= 6; k++) { // 状态计算
                if (j > k) dp[i][j] += dp[i - 1][j - k];
            }
        }
    }
    int total = pow(6, n); // 总的可能数
    vector<double> res;
    // 计算概率
    for (int i = n; i <= 6 * n; i++) {
        res.push_back(dp[n][i] * 1.0 / total);
    }
    return res;
}
```

## 扑克牌中的顺子(简单)

从扑克牌中随机抽5张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。[扑克牌中的顺子](https://leetcode-cn.com/problems/bu-ke-pai-zhong-de-shun-zi-lcof/)

### 参考代码

```c++
bool isStraight(vector<int>& nums) {
    // 先排序，然后统计零出现的次数，再统计数字构成连续的需要的间隔数
    sort(nums.begin(), nums.end());
    int nzeros = 0;
    for (auto n : nums) if (!n) nzeros++;
    int ngaps = 0;
    for (int i = nzeros, j = nzeros + 1; j < nums.size(); i++, j++) {
        if (nums[i] == nums[j]) return false;
        ngaps += nums[j] - nums[i] - 1;
    }
    return ngaps <= nzeros;
}
```

## 圆圈中最后剩下的数字(中等) 

0,1,,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字。求出这个圆圈里剩下的最后一个数字。[圆圈中最后剩下的数字](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)

### 参考代码

```c++
int lastRemaining(int n, int m) { // 模拟法
    if (n < 1 || m < 1) return -1;
    list<int> nums; // 构建循环链表
    for (int i = 0; i < n; i++) list.push_back(i);
    list<int>::iterator cur = nums.begin(); // 循环链表迭代器
    while (nums.size() > 1) {
        // 寻找第m个元素
        for (int i = 1; i < m; i++) {
            cur++;
            // 如果到链表尾的话，设置为链表头
            if (cur == nums.end()) cur = nums.begin();
        }
        // 记录下一个元素
        list<int>::iterator next = ++cur;
        if (next == nums.end()) next = nums.begin();
        cur--; // 恢复指针
        nums.erase(cur); // 当前元素出队
        cur = next; // 从下一个开始重新计数
    }
    return *cur;
}

/**
 * 数学法
 */
int lastRemaining(int n, int m) {
    int res = 0;
    for (int i = 2; i <= n; i++) {
        res = (res + m) % i;
    }
    return res;
}
```

## 股票的最大利润(中等)

假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？[股票的最大利润](https://leetcode-cn.com/problems/gu-piao-de-zui-da-li-run-lcof/)

### 参考代码

```c++
int maxProfit(vector<int>& prices) {
    // dp_i_0，表示第i填未持有股票，dp_i_1表示第i填持有股票
    int dp_i_0 = 0, dp_i_1 = INT_MIN;
    for (int i = 0; i < prices.size(); i++) {
        dp_i_0 = max(dp_i_0, dp_i_1 + prices[i]);
        dp_i_1 = max(dp_i_1, -prices[i]);
    }
    return dp_i_0;
}
```

##  求1+2+…+n(中等)

求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。[求1+2+…+n](https://leetcode-cn.com/problems/qiu-12n-lcof/)

### 参考代码

```c++
int sumNums(int n) {
    int res = n;
    bool t = (n > 0) && (res += sumNums(n - 1)) > 0;
    return res;
}
```

## 不用加减乘除做加法(简单)

写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。[不用加减乘除做加法](https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/)

### 参考代码

```c++
int add(int a, int b) {
    int res = 0, carry = 0;
    res = a ^ b;
    carry = (unsigned int) (a & b) << 1;
    int n1 = res, n2 = carry;
    while (n2) {
        res = n1 ^ n2;
        carry = (unsigned int)(n1 & n2) << 1;
        n1 = res, n2 = carry;
    }
    return res;
}
```

## 构建乘积数组(简单)

给定一个数组 A[0,1,…,n-1]，请构建一个数组 B[0,1,…,n-1]，其中 B 中的元素 `B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]`。不能使用除法。[构建乘积数组](https://leetcode-cn.com/problems/gou-jian-cheng-ji-shu-zu-lcof/)

### 参考代码

```c++
vector<int> constructArr(vector<int>& a) {
    int n = a.size();
    vector<int> res(n ,1);
    int t = 1;
    // 从前向后乘
    for (int i = 1; i < n; i++) {
        t *= a[i - 1];
        res[i] *= t;
    }
    t = 1;
    // 从后向前乘
    for (int i = n - 2; i >= 0; i--) {
        t *= a[i + 1];
        res[i] *= t;
    }
    return res;
}
```

## 把字符串转换成整数(中等)

写一个函数 StrToInt，实现把字符串转换成整数这个功能。不能使用 atoi 或者其他类似的库函数。[把字符串转换成整数](https://leetcode-cn.com/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof/)

### 参考代码

```c++
int strToInt(string s) {
    int idx = 0, res = 0;
    while (s[idx] == ' ') idx++;
    int sign = 1;
    if (s[idx] == '+') idx++;
    else if (s[idx] == '-') {
        sign = -1;
        idx++;
    }
    while (idx < s.size()) {
        if (s[idx] < '0' || s[idx] > '9') break;
        int t = sign * (s[idx] - '0');
        if (res > INT_MAX / 10 || (res == INT_MAX / 10 && t > 7)) return INT_MAX;
        if (res < INT_MIN / 10 || (res == INT_MIN / 10 && t < -8)) return INT_MIN;
        res = res * 10 + t;
        idx++;
    }
    return res;
}
```

## 二叉搜索树的最近公共祖先(简单)

给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。[二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

### 参考代码

```c++
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (!root || root == p || root == q) return root;
    if (root->val < p->val && root->val < q->val)
        return lowestCommonAncestor(root->right, p, q);
    if (root->val > p->val && root->val > q->val)
        return lowestCommonAncestor(root->left, p, q);
    return root;
}
```

## 二叉树的最近公共祖先(简单)

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。[二叉树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

### 参考代码

```c++
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (!root || root == p || root == q) return root;
    auto left = lowestCommonAncestor(root->left, p, q);
    auto right = lowestCommonAncestor(root->right, p, q);
    if (!left) return right;
    if (!right) return left;
    return root; 
}
```