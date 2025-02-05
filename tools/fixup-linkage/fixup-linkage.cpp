/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// The fixup-linkage tool is used to rewrite the LLVM IR produced by clang for
/// the classical compute code such that it can be linked correctly with the
/// LLVM IR that is generated for the quantum code. This avoids linker errors
/// such as "duplicate symbol definition".

#include <fstream>
#include <iostream>
#include <regex>

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cerr << "Usage:\n\tfixup-linkage <Quake-file> <LLVM-file> <output>\n";
    return 1;
  }

  // 1. Look for all the mangled kernel names. These will be found in the
  // mangled_name_map in the quake file. Add these names to `funcs`.
  std::ifstream modFile(argv[1]);
  std::string line;
  std::vector<std::string> funcs;
  {
    std::regex mapRegex{"quake\\.mangled_name_map = [{]"};
    std::regex stringRegex{"\"(.*?)\""};
    while (std::getline(modFile, line) && funcs.empty()) {
      auto funcsBegin =
          std::sregex_iterator(line.begin(), line.end(), mapRegex);
      auto rgxEnd = std::sregex_iterator();
      if (funcsBegin == rgxEnd)
        continue;
      auto names = line.substr(funcsBegin->str().size() - 1);
      auto namesBegin =
          std::sregex_iterator(names.begin(), names.end(), stringRegex);
      for (std::sregex_iterator i = namesBegin; i != rgxEnd; ++i) {
        auto s = i->str();
        funcs.push_back(s.substr(1, s.size() - 2));
      }
    }
    modFile.close();
    if (funcs.empty()) {
      std::cerr << "No mangled name map in the quake file.\n";
      return 1;
    }
  }

  // 2. Scan the LLVM file looking for the mangled kernel names. Where these
  // kernels are defined, they have their linkage modified to `linkonce_odr` if
  // that is not already the linkage. This change will prevent the duplicate
  // symbols defined error from the linker.
  std::ifstream llFile(argv[2]);
  std::ofstream outFile(argv[3]);
  std::regex filterRegex("^define ");
  std::regex filterInternalRegex("^define internal ");
  std::regex filterDsoLocalRegex("^define dso_local ");
  auto rgxEnd = std::sregex_iterator();
  auto computeCutPosition =
      [&](const std::string &matchStr) -> std::pair<bool, std::size_t> {
    std::regex rex("^" + matchStr + " ");
    auto iter = std::sregex_iterator(line.begin(), line.end(), rex);
    if (iter == rgxEnd)
      return {false, 0};
    return {true, matchStr.size()};
  };
  while (std::getline(llFile, line)) {
    auto iter = std::sregex_iterator(line.begin(), line.end(), filterRegex);
    if (iter == rgxEnd) {
      outFile << line << std::endl;
      continue;
    }
    if (line.find(" linkonce_odr ") != std::string::npos ||
        line.find(" weak dso_local ") != std::string::npos) {
      outFile << line << std::endl;
      continue;
    }
    // At this point, `line` starts with define but does not contain
    // linkonce_odr. So it is a candidate for being rewritten.
    bool replaced = false;
    for (auto fn : funcs) {
      // Check if this is defining one of our kernels.
      auto pos = line.find(fn);
      if (pos == std::string::npos)
        continue;
      auto pair = computeCutPosition("define internal");
      if (!pair.first) {
        pair = computeCutPosition("define dso_local");
        if (!pair.first) {
          pair = computeCutPosition("define");
          if (!pair.first) {
            // This is a hard error because the line must have a define.
            std::cerr << "internal error: line no longer matches.\n";
            return 1;
          }
        }
      }
      pos = pair.second;
      outFile << "define linkonce_odr dso_preemptable" << line.substr(pos)
              << std::endl;
      replaced = true;
      break;
    }
    if (!replaced)
      outFile << line << std::endl;
  }
  llFile.close();
  outFile.close();
  return 0;
}
