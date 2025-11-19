#include <vector>
#include <iostream>

void f() noexcept [[clang::nonblocking]]
{
  std::vector<int> v;
  v.push_back(42);
}

auto main() -> int
{
#if defined(__has_feature) && __has_feature(realtime_sanitizer)
  std::cout << "void f() noexcept [[clang::nonblocking]] Enabled" << std::endl;
#endif
  f();
  return 0;
}