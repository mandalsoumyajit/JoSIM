// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)

#include "JoSIM/Output.hpp"

#include "JoSIM/AnalysisType.hpp"
#include "JoSIM/Constants.hpp"
#include "JoSIM/Errors.hpp"
#include "JoSIM/FileOutputType.hpp"
#include "JoSIM/Input.hpp"
#include "JoSIM/ProgressBar.hpp"
#include "JoSIM/Simulation.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace JoSIM;

namespace {
struct StreamTraceDesc {
    std::string   name;
    char          type      = 'T';
    int64_t       fileIndex = -1;
    int           mode      = 0; // 0 = time, 1 = diff(v1 - v2), 2 = direct(v1), 3 = bdf2_volt(v1 - v2), 4 = isrc
    const double* v1        = nullptr;
    const double* v2        = nullptr;
    int64_t       srcIdx    = -1;
};

void eval_stream_col_range(const StreamTraceDesc& col,
                           const int64_t*         resIdx,
                           const double*          tPtr,
                           double                 cBDF2,
                           Matrix&                mObj,
                           size_t                 rStart,
                           size_t                 count,
                           double*                out,
                           size_t                 outStride) {
    if (col.mode == 0) {
        for (size_t r = 0; r < count; ++r) { out[r * outStride] = tPtr[resIdx[rStart + r]]; }
    } else if (col.mode == 1) {
        const double* p1 = col.v1;
        const double* p2 = col.v2;
        if (p1 && !p2) {
            for (size_t r = 0; r < count; ++r) { out[r * outStride] = p1[resIdx[rStart + r]]; }
        } else if (p1 && p2) {
            for (size_t r = 0; r < count; ++r) {
                const int64_t j    = resIdx[rStart + r];
                out[r * outStride] = p1[j] - p2[j];
            }
        } else if (!p1 && p2) {
            for (size_t r = 0; r < count; ++r) { out[r * outStride] = -p2[resIdx[rStart + r]]; }
        } else {
            for (size_t r = 0; r < count; ++r) { out[r * outStride] = 0.0; }
        }
    } else if (col.mode == 2) {
        const double* p1 = col.v1;
        if (p1) {
            for (size_t r = 0; r < count; ++r) { out[r * outStride] = p1[resIdx[rStart + r]]; }
        } else {
            for (size_t r = 0; r < count; ++r) { out[r * outStride] = 0.0; }
        }
    } else if (col.mode == 3) {
        const double* p1 = col.v1;
        const double* p2 = col.v2;
        if (p1 && !p2) {
            for (size_t r = 0; r < count; ++r) {
                const int64_t j  = resIdx[rStart + r];
                const double  d0 = p1[j];
                double        d1, d2;
                if (j >= 2) {
                    d1 = p1[j - 1];
                    d2 = p1[j - 2];
                } else if (j == 1) {
                    d2 = d1 = p1[0];
                } else {
                    d2 = d1 = d0;
                }
                out[r * outStride] = cBDF2 * (d0 - (4.0 / 3.0) * d1 + (1.0 / 3.0) * d2);
            }
        } else {
            for (size_t r = 0; r < count; ++r) {
                const int64_t j      = resIdx[rStart + r];
                const double  valin1 = p1 ? p1[j] : 0.0;
                const double  valin2 = p2 ? p2[j] : 0.0;
                double        valin1n1, valin1n2, valin2n1, valin2n2;
                if (j >= 2) {
                    valin1n2 = p1 ? p1[j - 2] : 0.0;
                    valin1n1 = p1 ? p1[j - 1] : 0.0;
                    valin2n2 = p2 ? p2[j - 2] : 0.0;
                    valin2n1 = p2 ? p2[j - 1] : 0.0;
                } else if (j == 1) {
                    valin1n2 = valin1n1 = p1 ? p1[0] : 0.0;
                    valin2n2 = valin2n1 = p2 ? p2[0] : 0.0;
                } else {
                    valin1n2 = valin1n1 = valin1;
                    valin2n2 = valin2n1 = valin2;
                }
                out[r * outStride] = cBDF2
                                     * ((valin1 - valin2) - (4.0 / 3.0) * (valin1n1 - valin2n1)
                                        + (1.0 / 3.0) * (valin1n2 - valin2n2));
            }
        }
    } else if (col.mode == 4) {
        Function localSrc = mObj.sourcegen.at(col.srcIdx);
        for (size_t r = 0; r < count; ++r) { out[r * outStride] = localSrc.value(tPtr[resIdx[rStart + r]]); }
    }
}

void write_streamed_row_major(std::ofstream&                        outfile,
                              const std::vector<StreamTraceDesc>&   cols,
                              const std::vector<int64_t>&           tIndices,
                              const std::vector<int64_t>&           result_indices,
                              const std::vector<double>&            t,
                              double                                cBDF2,
                              Matrix&                               mObj,
                              bool                                  useOpenMP) {
    const size_t numCols = tIndices.size();
    const size_t numRows = result_indices.size();
    if (numCols == 0 || numRows == 0) { return; }

    constexpr size_t R_TILE             = 64;
    constexpr size_t C_TILE             = 8;
    const size_t     targetChunkDoubles = 3 * 1024 * 1024; // ~24 MB chunk buffer
    const size_t     rawRowsPerChunk    = std::max<size_t>(R_TILE, targetChunkDoubles / std::max<size_t>(1, numCols));
    const size_t     rowsPerChunk       = std::min<size_t>(1024, ((rawRowsPerChunk + R_TILE - 1) / R_TILE) * R_TILE);
    std::vector<double> chunkBuf(rowsPerChunk * numCols);

    for (size_t rowStart = 0; rowStart < numRows; rowStart += rowsPerChunk) {
        const size_t rowEnd  = std::min(numRows, rowStart + rowsPerChunk);
        const size_t count   = rowEnd - rowStart;
        const size_t nRTiles = (count + R_TILE - 1) / R_TILE;
        const size_t nCTiles = (numCols + C_TILE - 1) / C_TILE;

#ifdef _OPENMP
        if (useOpenMP && count * numCols >= 32768) {
            const int nThreads = std::min<int>(16, omp_get_max_threads());
#pragma omp parallel for schedule(static) num_threads(nThreads)
            for (int64_t rtIdx = 0; rtIdx < static_cast<int64_t>(nRTiles); ++rtIdx) {
                const size_t rt   = static_cast<size_t>(rtIdx);
                const size_t r0   = rt * R_TILE;
                const size_t rCnt = std::min(R_TILE, count - r0);
                double       tile[R_TILE * C_TILE];
                for (size_t ct = 0; ct < nCTiles; ++ct) {
                    const size_t c0   = ct * C_TILE;
                    const size_t cCnt = std::min(C_TILE, numCols - c0);
                    for (size_t dc = 0; dc < cCnt; ++dc) {
                        eval_stream_col_range(cols[tIndices[c0 + dc]],
                                              result_indices.data(),
                                              t.data(),
                                              cBDF2,
                                              mObj,
                                              rowStart + r0,
                                              rCnt,
                                              tile + dc,
                                              C_TILE);
                    }
                    for (size_t r = 0; r < rCnt; ++r) {
                        std::copy_n(&tile[r * C_TILE], cCnt, &chunkBuf[(r0 + r) * numCols + c0]);
                    }
                }
            }
        } else
#endif
        {
            double tile[R_TILE * C_TILE];
            for (size_t rt = 0; rt < nRTiles; ++rt) {
                const size_t r0   = rt * R_TILE;
                const size_t rCnt = std::min(R_TILE, count - r0);
                for (size_t ct = 0; ct < nCTiles; ++ct) {
                    const size_t c0   = ct * C_TILE;
                    const size_t cCnt = std::min(C_TILE, numCols - c0);
                    for (size_t dc = 0; dc < cCnt; ++dc) {
                        eval_stream_col_range(cols[tIndices[c0 + dc]],
                                              result_indices.data(),
                                              t.data(),
                                              cBDF2,
                                              mObj,
                                              rowStart + r0,
                                              rCnt,
                                              tile + dc,
                                              C_TILE);
                    }
                    for (size_t r = 0; r < rCnt; ++r) {
                        std::copy_n(&tile[r * C_TILE], cCnt, &chunkBuf[(r0 + r) * numCols + c0]);
                    }
                }
            }
        }

        outfile.write(reinterpret_cast<const char*>(chunkBuf.data()),
                      static_cast<std::streamsize>(count * numCols * sizeof(double)));
    }
}
} // namespace

Output::Output(Input& iObj, Matrix& mObj, Simulation& sObj) {
    if (iObj.argEvents && iObj.eventsFile.has_value() && !iObj.eventsFile->empty()) {
        write_events(iObj, mObj, sObj);
    }
    if (iObj.argMargins && !iObj.marginTargets.empty()) {
        write_margins(iObj, mObj, sObj);
    }
    if (iObj.verifyEventsFile.has_value() && !iObj.verifyEventsFile->empty()) {
        verify_golden_events(iObj, mObj, sObj);
    }
    if (!iObj.assertions.empty()) {
        verify_assertions(iObj, mObj, sObj);
    }
    if (iObj.argCheckLoops) {
        write_loop_audit(iObj, mObj, sObj);
    }
    if (!iObj.jitterSpecs.empty()) {
        evaluate_jitter(iObj, mObj, sObj);
    }
    if (!iObj.measurements.empty()) {
        evaluate_measurements(iObj, mObj, sObj);
    } else if (iObj.argSens && !iObj.marginTargets.empty()) {
        write_sensitivity(iObj, mObj, sObj);
    }
    if (iObj.argEventsOnly
        || (!iObj.cli_output_file && iObj.output_files.empty() && sObj.results.mcRuns <= 1
            && (iObj.argMargins || iObj.argSens || iObj.verifyEventsFile.has_value() || !iObj.assertions.empty()
                || iObj.argCheckLoops || !iObj.jitterSpecs.empty() || !iObj.measurements.empty()))) {
        sObj.results.xVector.clear();
        sObj.results.xVector.shrink_to_fit();
        sObj.results.mcTraceOut.clear();
        sObj.results.mcTraceOut.shrink_to_fit();
        return;
    }

    bool hasUserRelevantTraces = false;
    for (const auto& tr : mObj.relevantTraces) {
        if (!tr.autoAddedForEvents) {
            hasUserRelevantTraces = true;
            break;
        }
    }

    bool allBinaryFiles = (iObj.cli_output_file.has_value() || !iObj.output_files.empty());
    auto is_bin_type    = [](FileOutputType ft) {
        return ft == FileOutputType::Npy || ft == FileOutputType::Bin || ft == FileOutputType::RawBin;
    };
    if (iObj.cli_output_file && !is_bin_type(iObj.cli_output_file->type())) { allBinaryFiles = false; }
    for (auto& f : iObj.output_files) {
        if (!is_bin_type(f.type())) { allBinaryFiles = false; }
    }

    bool canStreamDirect = allBinaryFiles && (sObj.results.mcRuns <= 1) && (iObj.transSim.prfirwindow() <= 0.0)
                           && !sObj.results.timeAxis.empty();
    if (canStreamDirect && hasUserRelevantTraces) {
        const size_t tSize = sObj.results.timeAxis.size();
        for (const auto& tr : mObj.relevantTraces) {
            if (tr.autoAddedForEvents) { continue; }
            const bool isJJ = (tr.deviceLabel.has_value() && tr.deviceLabel->size() > 3 && tr.deviceLabel->at(3) == 'B'
                               && tr.variableIndex.has_value());
            if (iObj.argAnal == AnalysisType::Voltage && tr.storageType == StorageType::Phase && !isJJ) {
                canStreamDirect = false;
                break;
            }
            if (tr.index1 && (!sObj.results.xVector[*tr.index1] || sObj.results.xVector[*tr.index1]->size() < tSize)) {
                canStreamDirect = false;
                break;
            }
            if (tr.index2 && (!sObj.results.xVector[*tr.index2] || sObj.results.xVector[*tr.index2]->size() < tSize)) {
                canStreamDirect = false;
                break;
            }
            if (tr.variableIndex
                && (!sObj.results.xVector[*tr.variableIndex]
                    || sObj.results.xVector[*tr.variableIndex]->size() < tSize)) {
                canStreamDirect = false;
                break;
            }
        }
    }

    if (canStreamDirect) {
        const auto&          x    = sObj.results.xVector;
        const auto&          t    = sObj.results.timeAxis;
        const auto&          tran = iObj.transSim;
        std::vector<int64_t> result_indices;
        if (!sObj.results.printIndices.empty()) {
            result_indices = sObj.results.printIndices;
        } else {
            for (size_t i = 0; i < t.size(); ++i) {
                if (t[i] >= tran.prstart()) {
                    result_indices.emplace_back(static_cast<int64_t>(i));
                    break;
                }
            }
            double next_print_point = tran.prstart() + tran.prstep();
            if (!result_indices.empty()) {
                for (size_t i = static_cast<size_t>(result_indices.back()); i < t.size(); ++i) {
                    if (t[i] >= next_print_point) {
                        result_indices.emplace_back(static_cast<int64_t>(i));
                        next_print_point += tran.prstep();
                    }
                }
            }
        }

        std::vector<StreamTraceDesc> cols;
        cols.reserve(1 + (hasUserRelevantTraces ? mObj.relevantTraces.size() : mObj.nm.size()));
        cols.push_back({"time", 'T', -1, 0, nullptr, nullptr, -1});
        traces.emplace_back("time");
        traces.back().type_ = 'T';

        if (hasUserRelevantTraces) {
            for (const auto& i : mObj.relevantTraces) {
                if (i.autoAddedForEvents) { continue; }
                StreamTraceDesc d;
                d.name      = i.deviceLabel.value();
                d.fileIndex = i.fIndex;
                d.v1        = i.index1 ? x[*i.index1]->data() : nullptr;
                d.v2        = i.index2 ? x[*i.index2]->data() : nullptr;
                const double* vv = i.variableIndex ? x[*i.variableIndex]->data() : nullptr;
                const bool    isJJ = (d.name.size() > 3 && d.name[3] == 'B' && vv != nullptr);
                if (i.storageType == StorageType::Voltage) {
                    d.type = 'V';
                    if (iObj.argAnal == AnalysisType::Voltage) {
                        d.mode = 1;
                    } else if (isJJ) {
                        d.mode = 2;
                        d.v1   = vv;
                    } else {
                        d.mode = 3;
                    }
                } else if (i.storageType == StorageType::Phase) {
                    d.type = 'P';
                    if (iObj.argAnal == AnalysisType::Phase) {
                        d.mode = 1;
                    } else {
                        d.mode = 2;
                        d.v1   = vv;
                    }
                } else {
                    d.type = 'I';
                    if (d.name.size() > 3 && d.name[3] == 'I') {
                        d.mode   = 4;
                        d.srcIdx = i.sourceIndex.value();
                    } else {
                        d.mode = 2;
                    }
                }
                cols.push_back(d);
                traces.emplace_back(d.name);
                traces.back().type_     = d.type;
                traces.back().fileIndex = d.fileIndex;
            }
        } else {
            for (const auto& nmEntry : mObj.nm) {
                StreamTraceDesc d;
                d.name      = (iObj.argAnal == AnalysisType::Voltage ? "V(" : "P(") + nmEntry.first + ")";
                d.type      = (iObj.argAnal == AnalysisType::Voltage) ? 'V' : 'P';
                d.fileIndex = -1;
                d.mode      = 2;
                d.v1        = x[nmEntry.second] ? x[nmEntry.second]->data() : nullptr;
                cols.push_back(d);
                traces.emplace_back(d.name);
                traces.back().type_ = d.type;
            }
        }

        const double cBDF2     = (3.0 * Constants::SIGMA) / (2.0 * tran.tstep());
        const bool   useOpenMP = (iObj.argPar || iObj.argGpu);

        auto stream_one_file = [&](const std::string& filename, FileOutputType outType, int64_t fIndex) {
            std::vector<int64_t> tIndices = {0};
            for (size_t i = 1; i < cols.size(); ++i) {
                if (cols[i].fileIndex == fIndex || fIndex == -1) { tIndices.emplace_back(static_cast<int64_t>(i)); }
            }
            std::ofstream     outfile(filename, std::ios::out | std::ios::binary);
            std::vector<char> ioBuf(4 * 1024 * 1024);
            outfile.rdbuf()->pubsetbuf(ioBuf.data(), static_cast<std::streamsize>(ioBuf.size()));
            if (!outfile.is_open()) {
                Errors::output_errors(OutputErrors::CANNOT_OPEN_FILE, filename);
                return;
            }
            if (cols.empty()) {
                Errors::output_errors(OutputErrors::NOTHING_SPECIFIED);
                return;
            }
            const size_t numCols = tIndices.size();
            const size_t numRows = result_indices.size();

            if (outType == FileOutputType::Npy) {
                std::string dict = "{'descr': [";
                for (size_t c = 0; c < numCols; ++c) {
                    std::string name = cols[tIndices[c]].name;
                    if (name.size() >= 2 && name.front() == '"' && name.back() == '"') {
                        name = name.substr(1, name.size() - 2);
                    }
                    std::string escaped;
                    escaped.reserve(name.size());
                    for (char ch : name) {
                        if (ch == '\'' || ch == '\\') { escaped.push_back('\\'); }
                        escaped.push_back(ch);
                    }
                    dict += "('" + escaped + "', '<f8')";
                    if (c + 1 < numCols) { dict += ", "; }
                }
                dict += "], 'fortran_order': False, 'shape': (" + std::to_string(numRows) + ",), }";

                const bool   useV2     = (10 + dict.size() + 1 > 65535);
                const size_t prefixLen = useV2 ? 12 : 10;
                const size_t rem       = (prefixLen + dict.size() + 1) % 64;
                const size_t padLen    = (rem == 0) ? 0 : (64 - rem);
                dict.append(padLen, ' ');
                dict.push_back('\n');

                const char magic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
                outfile.write(magic, 6);
                if (!useV2) {
                    const char     ver[2]  = {'\x01', '\x00'};
                    const uint16_t hdrSize = static_cast<uint16_t>(dict.size());
                    outfile.write(ver, 2);
                    outfile.write(reinterpret_cast<const char*>(&hdrSize), sizeof(uint16_t));
                } else {
                    const char     ver[2]  = {'\x02', '\x00'};
                    const uint32_t hdrSize = static_cast<uint32_t>(dict.size());
                    outfile.write(ver, 2);
                    outfile.write(reinterpret_cast<const char*>(&hdrSize), sizeof(uint32_t));
                }
                outfile.write(dict.data(), static_cast<std::streamsize>(dict.size()));

                write_streamed_row_major(outfile, cols, tIndices, result_indices, t, cBDF2, mObj, useOpenMP);
            } else if (outType == FileOutputType::RawBin) {
                outfile << "Title: " << std::filesystem::path(filename).stem().string() << "\n";
                std::time_t result = std::time(nullptr);
                outfile << "Date: " << std::asctime(std::localtime(&result));
                outfile << "Plotname: Transient Analysis\n";
                outfile << "Flags: real\n";
                outfile << "No. Variables: " << numCols << "\n";
                outfile << "No. Points: " << numRows << "\n";
                outfile << "Command: JoSIM v" << VERSION << "\n";
                outfile << "Variables:\n";
                outfile << " 0 time Seconds\n";
                for (size_t i = 1; i < numCols; ++i) {
                    std::string name = cols[tIndices[i]].name;
                    name.erase(std::remove(name.begin(), name.end(), '\"'), name.end());
                    const char tc = cols[tIndices[i]].type;
                    if (tc == 'V') {
                        outfile << " " << i << " " << name << " Voltage\n";
                    } else if (tc == 'P') {
                        outfile << " " << i << " " << name << " Phase\n";
                    } else if (tc == 'I') {
                        std::replace(name.begin(), name.end(), '|', '.');
                        if (name.size() > 3) { name = name.substr(2, name.size() - 3); }
                        outfile << " " << i << " " << name << "#branch Current\n";
                    }
                }
                outfile << "Binary:\n";

                write_streamed_row_major(outfile, cols, tIndices, result_indices, t, cBDF2, mObj, useOpenMP);
            } else if (outType == FileOutputType::Bin) {
                const uint64_t nC       = static_cast<uint64_t>(numCols);
                const uint64_t nR       = static_cast<uint64_t>(numRows);
                const uint32_t version  = 1;
                const uint32_t flags    = 0;
                const char     magic[8] = {'J', 'O', 'S', 'I', 'M', 'B', 'I', 'N'};

                outfile.write(magic, 8);
                outfile.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));
                outfile.write(reinterpret_cast<const char*>(&flags), sizeof(uint32_t));
                outfile.write(reinterpret_cast<const char*>(&nC), sizeof(uint64_t));
                outfile.write(reinterpret_cast<const char*>(&nR), sizeof(uint64_t));

                for (size_t c = 0; c < numCols; ++c) {
                    const auto&    tr      = cols[tIndices[c]];
                    const uint32_t nameLen = static_cast<uint32_t>(tr.name.size());
                    outfile.write(reinterpret_cast<const char*>(&nameLen), sizeof(uint32_t));
                    outfile.write(tr.name.data(), nameLen);
                    outfile.write(&tr.type, 1);
                }

                std::vector<double>   colBuf(numRows);
                const std::streamsize colBytes = static_cast<std::streamsize>(numRows * sizeof(double));
                for (size_t c = 0; c < numCols; ++c) {
                    eval_stream_col_range(
                            cols[tIndices[c]], result_indices.data(), t.data(), cBDF2, mObj, 0, numRows, colBuf.data(), 1);
                    outfile.write(reinterpret_cast<const char*>(colBuf.data()), colBytes);
                }
            }
        };

        if (iObj.cli_output_file) {
            stream_one_file(iObj.cli_output_file->name(), iObj.cli_output_file->type(), -1);
        }
        if (!iObj.output_files.empty()) {
            for (size_t i = 0; i < iObj.output_files.size(); ++i) {
                stream_one_file(iObj.output_files[i].name(), iObj.output_files[i].type(), static_cast<int64_t>(i));
            }
        }
        sObj.results.xVector.clear();
        sObj.results.xVector.shrink_to_fit();
        return;
    }

    // Write the output in type agnostic format
    write_output(iObj, mObj, sObj);
    sObj.results.xVector.clear();
    sObj.results.xVector.shrink_to_fit();
    // Format the output into the relevant type
    if (iObj.cli_output_file) {
        const auto outType = iObj.cli_output_file.value().type();
        const auto outName = iObj.cli_output_file.value().name();
        if (outType == FileOutputType::Csv) {
            format_csv_or_dat(outName, ',', iObj.argMin);
        } else if (outType == FileOutputType::Dat) {
            format_csv_or_dat(outName, ' ', iObj.argMin);
        } else if (outType == FileOutputType::Raw) {
            format_raw(outName, iObj.argMin);
        } else if (outType == FileOutputType::Bin) {
            format_bin(outName, iObj.argMin);
        } else if (outType == FileOutputType::Npy) {
            format_npy(outName, iObj.argMin);
        } else if (outType == FileOutputType::RawBin) {
            format_raw_bin(outName, iObj.argMin);
        }
    }
    if (!iObj.output_files.empty()) {
        for (auto i = 0; i < iObj.output_files.size(); ++i) {
            const auto outType = iObj.output_files.at(i).type();
            const auto outName = iObj.output_files.at(i).name();
            if (outType == FileOutputType::Csv) {
                format_csv_or_dat(outName, ',', iObj.argMin, i);
            } else if (outType == FileOutputType::Dat) {
                format_csv_or_dat(outName, ' ', iObj.argMin, i);
            } else if (outType == FileOutputType::Raw) {
                format_raw(outName, iObj.argMin, i);
            } else if (outType == FileOutputType::Bin) {
                format_bin(outName, iObj.argMin, i);
            } else if (outType == FileOutputType::Npy) {
                format_npy(outName, iObj.argMin, i);
            } else if (outType == FileOutputType::RawBin) {
                format_raw_bin(outName, iObj.argMin, i);
            }
        }
    }
    if (!iObj.cli_output_file && iObj.output_files.empty() && !iObj.argMargins && !iObj.argSens && !iObj.argSweep
        && !iObj.verifyEventsFile.has_value() && iObj.assertions.empty() && iObj.jitterSpecs.empty()
        && iObj.measurements.empty()) {
        format_cout(iObj.argMin);
    }
    sObj.results.mcTraceOut.clear();
    sObj.results.mcTraceOut.shrink_to_fit();
}

void Output::write_output(const Input& iObj, Matrix& mObj, Simulation& sObj) {
    // Shorthand
    auto&   x           = sObj.results.xVector;
    auto&   t           = sObj.results.timeAxis;
    auto&   tran        = iObj.transSim;
    // Create downsampling FIR window, use Hanning function with odd number of taps.
    // The FIR filter is centered around the requested point in time, i. e. looks into the past and future.
    int64_t halffirsize = static_cast<int64_t>(tran.prfirwindow() <= 0. ? 0 : .5 * tran.prfirwindow() / tran.tstep());
    int64_t firsize     = halffirsize * 2 + 1; // Guarantees odd number of taps or 1 in case of disabled filter.
    std::vector<double> firwindow;
    firwindow.reserve(firsize);
    // Calculate Hanning function 1 - cos(x) with x in ]0; 2*pi[.
    // For firsize == 1 (filter disabled), this results in a firwindow = {1.0},
    //   i. e. a convolution with delta(x), which has no effect.
    double firsum = 0.0;
    for (int64_t k = 0; k < firsize; k++) {
        double hanning  = 1. - cos(((double) k + 0.5) * (2. * Constants::PI / (double) firsize));
        firsum         += hanning;
        firwindow.emplace_back(hanning);
    }
    double firscale = 1. / firsum; // Normalize Hanning function to sum(firwindow) = 1
    for (auto& hanning : firwindow) { hanning = firscale * hanning; }
    // Indices to print
    std::vector<int64_t> result_indices;
    if (!sObj.results.printIndices.empty()) {
        result_indices = sObj.results.printIndices;
    } else {
        for (auto i = 0; i < t.size(); ++i) {
            if (t.at(i) >= tran.prstart()) {
                result_indices.emplace_back(i);
                break;
            }
        }
        double next_print_point = tran.prstart() + tran.prstep();
        if (!result_indices.empty()) {
            for (auto i = static_cast<size_t>(result_indices.back()); i < t.size(); ++i) {
                if (t.at(i) >= next_print_point) {
                    result_indices.emplace_back(i);
                    next_print_point += tran.prstep();
                }
            }
        }
    }
    // Create the time trace
    traces.emplace_back("time");
    traces.back().type_ = 'T';
    traces.back().data_.reserve(result_indices.size());
    for (auto i : result_indices) { traces.back().data_.emplace_back(t.at(i)); }

    std::vector<int64_t> userTraceIndices;
    userTraceIndices.reserve(mObj.relevantTraces.size());
    for (size_t idx = 0; idx < mObj.relevantTraces.size(); ++idx) {
        if (!mObj.relevantTraces[idx].autoAddedForEvents) {
            userTraceIndices.push_back(static_cast<int64_t>(idx));
        }
    }
    const bool hasUserTraces = !userTraceIndices.empty();

    // Print only the indices of the relevant traces
    int64_t cc = 0;
    if (mObj.relevantTraces.size() != 0 && sObj.results.mcRuns > 1 && !iObj.argMargins && !iObj.argSens
        && !sObj.results.mcTraceOut.empty()) {
        const int64_t B          = sObj.results.mcRuns;
        const int64_t nRows      = sObj.results.mcNumRows;
        const int64_t nKept      = sObj.results.mcNumKeptSteps;
        const int64_t nTraces    = static_cast<int64_t>(mObj.relevantTraces.size());
        const size_t  nPrint     = result_indices.size();

        auto make_mc_label = [](const std::string& label, const std::string& suffix) -> std::string {
            if (suffix.empty()) { return label; }
            if (label.size() >= 2 && label.front() == '"' && label.back() == '"') {
                return label.substr(0, label.size() - 1) + suffix + "\"";
            }
            return label + suffix;
        };

        std::vector<int64_t> traceBaseSlot(static_cast<size_t>(nTraces), -1);
        for (int64_t idx = 0; idx < nTraces; ++idx) {
            const auto& i = mObj.relevantTraces[idx];
            if (hasUserTraces && i.autoAddedForEvents) { continue; }
            traceBaseSlot[idx]      = static_cast<int64_t>(traces.size());
            const std::string label = i.deviceLabel.value();
            char              tChar = 'I';
            if (i.storageType == StorageType::Voltage) {
                tChar = 'V';
            } else if (i.storageType == StorageType::Phase) {
                tChar = 'P';
            }
            const char* suffixes[4] = {"", "_STD", "_MIN", "_MAX"};
            for (int s = 0; s < 4; ++s) {
                traces.emplace_back(make_mc_label(label, suffixes[s]));
                traces.back().fileIndex = i.fIndex;
                traces.back().type_     = tChar;
            }
            if (iObj.argMcAll) {
                for (int64_t b = 0; b < B; ++b) {
                    traces.emplace_back(make_mc_label(label, "_MC" + std::to_string(b)));
                    traces.back().fileIndex = i.fIndex;
                    traces.back().type_     = tChar;
                }
            }
        }

        const bool isIdentityPrint = (firsize == 1 && nPrint == static_cast<size_t>(nKept)
                                      && (nPrint == 0
                                          || (result_indices.front() == 0
                                              && result_indices.back() == static_cast<int64_t>(nKept - 1))));

        struct McStat {
            double  peakMean = 0.0;
            double  peakStd  = 0.0;
            double  peakMin  = 0.0;
            double  peakMax  = 0.0;
            int64_t nomSfq   = 0;
            int64_t passCnt  = 0;
            int64_t bLow     = 0;
            int64_t bHigh    = 0;
            bool    isPhase  = false;
        };
        std::vector<McStat> stats(nTraces);

        auto process_mc_trace = [&](int64_t idx) {
            const auto&   i        = mObj.relevantTraces[idx];
            const auto&   st       = i.storageType;
            const int64_t baseSlot = traceBaseSlot[idx];
            std::vector<double> dummyMean, dummyStd, dummyMin, dummyMax;
            auto&         trMean   = (baseSlot >= 0) ? traces[baseSlot + 0].data_ : dummyMean;
            auto&         trStd    = (baseSlot >= 0) ? traces[baseSlot + 1].data_ : dummyStd;
            auto&         trMin    = (baseSlot >= 0) ? traces[baseSlot + 2].data_ : dummyMin;
            auto&         trMax    = (baseSlot >= 0) ? traces[baseSlot + 3].data_ : dummyMax;
            trMean.resize(nPrint, 0.0);
            trStd.resize(nPrint, 0.0);
            trMin.resize(nPrint, 0.0);
            trMax.resize(nPrint, 0.0);

            const int64_t i1   = i.index1.has_value() ? i.index1.value() : -1;
            const int64_t i2   = i.index2.has_value() ? i.index2.value() : -1;
            const int64_t vi   = i.variableIndex.has_value() ? i.variableIndex.value() : -1;
            const int32_t s1   = (i1 >= 0 && i1 < static_cast<int64_t>(sObj.results.mcRowToSlot.size()))
                                         ? sObj.results.mcRowToSlot[i1]
                                         : -1;
            const int32_t s2   = (i2 >= 0 && i2 < static_cast<int64_t>(sObj.results.mcRowToSlot.size()))
                                         ? sObj.results.mcRowToSlot[i2]
                                         : -1;
            const int32_t sv   = (vi >= 0 && vi < static_cast<int64_t>(sObj.results.mcRowToSlot.size()))
                                         ? sObj.results.mcRowToSlot[vi]
                                         : -1;
            const size_t xsize = std::min<size_t>(t.size(), static_cast<size_t>(nKept));

            Function localSrc;
            bool     isISrc = (st == StorageType::Current && i.deviceLabel.value().at(3) == 'I');
            if (isISrc) { localSrc = mObj.sourcegen.at(i.sourceIndex.value()); }

            int32_t directSlot = -1;
            if (isIdentityPrint) {
                if (st == StorageType::Phase && iObj.argAnal == AnalysisType::Phase && s1 >= 0 && s2 < 0) {
                    directSlot = s1;
                } else if (st == StorageType::Phase && iObj.argAnal == AnalysisType::Voltage
                           && i.deviceLabel.value().at(3) == 'B' && sv >= 0) {
                    directSlot = sv;
                } else if (st == StorageType::Voltage && iObj.argAnal == AnalysisType::Voltage && s1 >= 0 && s2 < 0) {
                    directSlot = s1;
                } else if (st == StorageType::Voltage && iObj.argAnal == AnalysisType::Phase
                           && i.deviceLabel.value().at(3) == 'B' && sv >= 0) {
                    directSlot = sv;
                } else if (st == StorageType::Current && !isISrc && s1 >= 0) {
                    directSlot = s1;
                }
            }

            std::vector<double>  meanVec(nPrint, 0.0);
            std::vector<double>  m2Vec(nPrint, 0.0);
            std::vector<double>  vMin(nPrint, 1e300);
            std::vector<double>  vMax(nPrint, -1e300);
            std::vector<double>  trialWave(directSlot >= 0 ? 0 : nPrint, 0.0);
            std::vector<int64_t> trialSfq(B, 0);
            std::vector<double>  vectphase;
            if (st == StorageType::Phase && iObj.argAnal == AnalysisType::Voltage
                && !(i.deviceLabel.value().at(3) == 'B' && sv >= 0)) {
                vectphase.resize(xsize, 0.0);
            }

            for (int64_t b = 0; b < B; ++b) {
                const double* p1 = (s1 >= 0) ? (sObj.results.mcTraceOut.data() + (b * nRows + s1) * nKept) : nullptr;
                const double* p2 = (s2 >= 0) ? (sObj.results.mcTraceOut.data() + (b * nRows + s2) * nKept) : nullptr;
                const double* pv = (sv >= 0) ? (sObj.results.mcTraceOut.data() + (b * nRows + sv) * nKept) : nullptr;

                const double* wavePtr = nullptr;
                if (directSlot >= 0) {
                    wavePtr = sObj.results.mcTraceOut.data() + (b * nRows + directSlot) * nKept;
                    if (iObj.argMcAll && baseSlot >= 0) { traces[baseSlot + 4 + b].extData_ = wavePtr; }
                } else {

                if (st == StorageType::Voltage) {
                    for (size_t pIdx = 0; pIdx < nPrint; ++pIdx) {
                        const int64_t j   = result_indices[pIdx];
                        double        acc = 0.0;
                        for (int64_t k = 0; k < firsize; ++k) {
                            int64_t jk = j + halffirsize - k;
                            if (jk < 0) {
                                jk = 0;
                            } else if (static_cast<size_t>(jk) >= xsize) {
                                jk = static_cast<int64_t>(xsize) - 1;
                            }
                            const double valin1 = p1 ? p1[jk] : 0.0;
                            const double valin2 = p2 ? p2[jk] : 0.0;
                            double       value  = 0.0;
                            if (iObj.argAnal == AnalysisType::Voltage) {
                                value = valin1 - valin2;
                            } else if (i.deviceLabel.value().at(3) == 'B' && pv) {
                                value = pv[jk];
                            } else {
                                double valin1n1, valin1n2, valin2n1, valin2n2;
                                if (jk >= 2) {
                                    valin1n2 = p1 ? p1[jk - 2] : 0.0;
                                    valin1n1 = p1 ? p1[jk - 1] : 0.0;
                                    valin2n2 = p2 ? p2[jk - 2] : 0.0;
                                    valin2n1 = p2 ? p2[jk - 1] : 0.0;
                                } else if (jk == 1) {
                                    valin1n2 = valin1n1 = p1 ? p1[jk - 1] : 0.0;
                                    valin2n2 = valin2n1 = p2 ? p2[jk - 1] : 0.0;
                                } else {
                                    valin1n2 = valin1n1 = valin1;
                                    valin2n2 = valin2n1 = valin2;
                                }
                                value = ((3.0 * Constants::SIGMA) / (2.0 * iObj.transSim.tstep()))
                                        * ((valin1 - valin2) - (4.0 / 3.0) * (valin1n1 - valin2n1)
                                           + (1.0 / 3.0) * (valin1n2 - valin2n2));
                            }
                            acc += firwindow[k] * value;
                        }
                        trialWave[pIdx] = acc;
                    }
                } else if (st == StorageType::Phase) {
                    if (iObj.argAnal == AnalysisType::Phase) {
                        for (size_t pIdx = 0; pIdx < nPrint; ++pIdx) {
                            const int64_t j   = result_indices[pIdx];
                            double        acc = 0.0;
                            for (int64_t k = 0; k < firsize; ++k) {
                                int64_t jk = j + halffirsize - k;
                                if (jk < 0) {
                                    jk = 0;
                                } else if (static_cast<size_t>(jk) >= xsize) {
                                    jk = static_cast<int64_t>(xsize) - 1;
                                }
                                acc += firwindow[k] * ((p1 ? p1[jk] : 0.0) - (p2 ? p2[jk] : 0.0));
                            }
                            trialWave[pIdx] = acc;
                        }
                    } else if (i.deviceLabel.value().at(3) == 'B' && pv) {
                        for (size_t pIdx = 0; pIdx < nPrint; ++pIdx) {
                            const int64_t j   = result_indices[pIdx];
                            double        acc = 0.0;
                            for (int64_t k = 0; k < firsize; ++k) {
                                int64_t jk = j + halffirsize - k;
                                if (jk < 0) {
                                    jk = 0;
                                } else if (static_cast<size_t>(jk) >= xsize) {
                                    jk = static_cast<int64_t>(xsize) - 1;
                                }
                                acc += firwindow[k] * pv[jk];
                            }
                            trialWave[pIdx] = acc;
                        }
                    } else {
                        double phaseN1 = 0.0, phaseN2 = 0.0;
                        for (size_t j = 0; j < xsize; ++j) {
                            const double valin1 = p1 ? p1[j] : 0.0;
                            const double valin2 = p2 ? p2[j] : 0.0;
                            const double value
                                    = ((2.0 * iObj.transSim.tstep()) / (3.0 * Constants::SIGMA)) * (valin1 - valin2)
                                      + (4.0 / 3.0) * phaseN1 - (1.0 / 3.0) * phaseN2;
                            phaseN2      = phaseN1;
                            phaseN1      = value;
                            vectphase[j] = value;
                        }
                        for (size_t pIdx = 0; pIdx < nPrint; ++pIdx) {
                            const int64_t j   = result_indices[pIdx];
                            double        acc = 0.0;
                            for (int64_t k = 0; k < firsize; ++k) {
                                int64_t jk = j + halffirsize - k;
                                if (jk < 0) {
                                    jk = 0;
                                } else if (static_cast<size_t>(jk) >= xsize) {
                                    jk = static_cast<int64_t>(xsize) - 1;
                                }
                                acc += firwindow[k] * vectphase[jk];
                            }
                            trialWave[pIdx] = acc;
                        }
                    }
                } else {
                    for (size_t pIdx = 0; pIdx < nPrint; ++pIdx) {
                        const int64_t j   = result_indices[pIdx];
                        double        acc = 0.0;
                        for (int64_t k = 0; k < firsize; ++k) {
                            int64_t jk = j + halffirsize - k;
                            if (jk < 0) {
                                jk = 0;
                            } else if (static_cast<size_t>(jk) >= xsize) {
                                jk = static_cast<int64_t>(xsize) - 1;
                            }
                            const double value = (!isISrc) ? (p1 ? p1[jk] : 0.0) : localSrc.value(t[jk]);
                            acc += firwindow[k] * value;
                        }
                        trialWave[pIdx] = acc;
                    }
                }
                wavePtr = trialWave.data();
                if (iObj.argMcAll && baseSlot >= 0) { traces[baseSlot + 4 + b].data_ = trialWave; }
                }

                if (st == StorageType::Phase && nPrint > 0) {
                    const double dPhi = wavePtr[nPrint - 1] - wavePtr[0];
                    trialSfq[b]       = static_cast<int64_t>(std::llround(dPhi / (2.0 * Constants::PI)));
                }
                const double invCount = 1.0 / static_cast<double>(b + 1);
                for (size_t pIdx = 0; pIdx < nPrint; ++pIdx) {
                    const double v      = wavePtr[pIdx];
                    const double delta  = v - meanVec[pIdx];
                    meanVec[pIdx]      += delta * invCount;
                    const double delta2 = v - meanVec[pIdx];
                    m2Vec[pIdx]        += delta * delta2;
                    if (v < vMin[pIdx]) { vMin[pIdx] = v; }
                    if (v > vMax[pIdx]) { vMax[pIdx] = v; }
                }
            }

            const double invB = 1.0 / static_cast<double>(B);
            size_t       pk   = 0;
            for (size_t pIdx = 0; pIdx < nPrint; ++pIdx) {
                const double mean = meanVec[pIdx];
                const double var  = std::max(0.0, m2Vec[pIdx] * invB);
                trMean[pIdx]      = mean;
                trStd[pIdx]       = std::sqrt(var);
                trMin[pIdx]       = vMin[pIdx];
                trMax[pIdx]       = vMax[pIdx];
                if (std::abs(mean) > std::abs(trMean[pk])) { pk = pIdx; }
            }
            if (nPrint > 0) {
                stats[idx].peakMean = trMean[pk];
                stats[idx].peakStd  = trStd[pk];
                stats[idx].peakMin  = trMin[pk];
                stats[idx].peakMax  = trMax[pk];
            }
            if (st == StorageType::Phase && nPrint > 0 && B > 0) {
                stats[idx].isPhase = true;
                int64_t bNom       = B / 2;
                if (iObj.argSweep && !iObj.sweeps.empty()) {
                    auto find_nom = [](const SweepDim& sd) -> int64_t {
                        if (sd.values.empty()) { return 0; }
                        if (sd.isRelative) {
                            int64_t best = 0;
                            for (size_t k = 1; k < sd.values.size(); ++k) {
                                if (std::abs(sd.values[k] - 1.0) < std::abs(sd.values[best] - 1.0)) {
                                    best = static_cast<int64_t>(k);
                                }
                            }
                            return best;
                        }
                        return static_cast<int64_t>(sd.values.size() / 2);
                    };
                    if (iObj.sweeps.size() == 1) {
                        bNom = std::clamp<int64_t>(find_nom(iObj.sweeps[0]), 0, B - 1);
                    } else if (iObj.sweeps.size() >= 2) {
                        const int64_t n1 = static_cast<int64_t>(iObj.sweeps[1].values.size());
                        bNom = std::clamp<int64_t>(find_nom(iObj.sweeps[0]) * n1 + find_nom(iObj.sweeps[1]), 0, B - 1);
                    }
                }
                const int64_t refSfq = trialSfq[bNom];
                stats[idx].nomSfq    = refSfq;
                int64_t passCnt      = 0;
                for (int64_t b = 0; b < B; ++b) {
                    if (trialSfq[b] == refSfq) { ++passCnt; }
                }
                stats[idx].passCnt = passCnt;
                int64_t lo         = bNom;
                int64_t hi         = bNom;
                while (lo > 0 && trialSfq[lo - 1] == refSfq) { --lo; }
                while (hi + 1 < B && trialSfq[hi + 1] == refSfq) { ++hi; }
                stats[idx].bLow  = lo;
                stats[idx].bHigh = hi;
            }
        };

#ifdef _OPENMP
        if ((iObj.argPar || iObj.argGpu) && nTraces >= 2) {
            const int nThreads = std::min<int>(omp_get_max_threads(), static_cast<int>(nTraces));
#pragma omp parallel for schedule(dynamic) num_threads(nThreads)
            for (int64_t idx = 0; idx < nTraces; ++idx) { process_mc_trace(idx); }
        } else
#endif
        {
            for (int64_t idx = 0; idx < nTraces; ++idx) { process_mc_trace(idx); }
        }

        if (!iObj.argMcAll) {
            sObj.results.mcTraceOut.clear();
            sObj.results.mcTraceOut.shrink_to_fit();
        }

        if (!iObj.argMin && nTraces > 0) {
            if (iObj.argSweep && !iObj.sweeps.empty()) {
                std::cout << "\nJoSIM Parameter Sweep / Margin Summary (" << B << " sweep points):\n";
                for (size_t d = 0; d < iObj.sweeps.size(); ++d) {
                    const auto& sd = iObj.sweeps[d];
                    if (!sd.values.empty()) {
                        std::cout << "  Dim " << d << " (" << sd.target << "): " << sd.values.size() << " pts ["
                                  << std::defaultfloat << std::setprecision(4) << sd.values.front() << " .. "
                                  << sd.values.back() << "]" << (sd.isRelative ? " (relative scale)" : "") << "\n";
                    }
                }
                const bool is1D = (iObj.sweeps.size() == 1 && static_cast<int64_t>(iObj.sweeps[0].values.size()) == B);
                std::cout << "  " << std::left << std::setw(22) << "Trace" << std::right << std::setw(14) << "Peak Mean"
                          << std::setw(14) << "Peak Min" << std::setw(14) << "Peak Max" << std::setw(10) << "Nom SFQ"
                          << std::setw(12) << "Pass Pts" << (is1D ? "  Operating Window" : "") << "\n";
                const int64_t maxShow = std::min<int64_t>(nTraces, 16);
                for (int64_t idx = 0; idx < maxShow; ++idx) {
                    const auto& st = stats[idx];
                    std::cout << "  " << std::left << std::setw(22) << mObj.relevantTraces[idx].deviceLabel.value()
                              << std::right << std::scientific << std::setprecision(4) << std::setw(14) << st.peakMean
                              << std::setw(14) << st.peakMin << std::setw(14) << st.peakMax;
                    if (st.isPhase) {
                        std::ostringstream passStr;
                        passStr << st.passCnt << "/" << B;
                        std::cout << std::setw(10) << st.nomSfq << std::setw(12) << passStr.str();
                        if (is1D) {
                            const auto&  vals = iObj.sweeps[0].values;
                            const double vLo  = vals[st.bLow];
                            const double vHi  = vals[st.bHigh];
                            std::cout << "  [" << std::defaultfloat << std::setprecision(4) << vLo << ", " << vHi
                                      << "]";
                            if (iObj.sweeps[0].isRelative) {
                                std::cout << " (" << std::showpos << std::fixed << std::setprecision(1)
                                          << (vLo - 1.0) * 100.0 << "%, " << (vHi - 1.0) * 100.0 << "%"
                                          << std::noshowpos << ")";
                            }
                        }
                    } else {
                        std::cout << std::setw(10) << "-" << std::setw(12) << "-";
                    }
                    std::cout << "\n";
                }
                if (nTraces > maxShow) {
                    std::cout << "  ... (" << (nTraces - maxShow) << " additional traces written to output)\n";
                }
                std::cout << "\n";
            } else {
                std::cout << "\nJoSIM Monte Carlo Robustness Summary (" << B << " trials):\n";
                std::cout << "  " << std::left << std::setw(22) << "Trace" << std::right << std::setw(14) << "Peak Mean"
                          << std::setw(14) << "Peak 1-Sigma" << std::setw(14) << "Peak Min" << std::setw(14)
                          << "Peak Max" << std::setw(12) << "CV (%)" << "\n";
                const int64_t maxShow = std::min<int64_t>(nTraces, 16);
                for (int64_t idx = 0; idx < maxShow; ++idx) {
                    const auto&  st = stats[idx];
                    const double cv = (std::abs(st.peakMean) > 1e-18) ? (100.0 * st.peakStd / std::abs(st.peakMean))
                                                                      : 0.0;
                    std::cout << "  " << std::left << std::setw(22) << mObj.relevantTraces[idx].deviceLabel.value()
                              << std::right << std::scientific << std::setprecision(4) << std::setw(14) << st.peakMean
                              << std::setw(14) << st.peakStd << std::setw(14) << st.peakMin << std::setw(14)
                              << st.peakMax << std::fixed << std::setprecision(2) << std::setw(11) << cv << "%\n";
                }
                if (nTraces > maxShow) {
                    std::cout << "  ... (" << (nTraces - maxShow) << " additional traces written to output)\n";
                }
                std::cout << "\n";
            }
        }
    } else if (hasUserTraces) {
        const int64_t nUserTraces = static_cast<int64_t>(userTraceIndices.size());
        // Create progress bar
        ProgressBar bar;
        if (!iObj.argMin) {
            bar.create_thread();
            bar.set_bar_width(30);
            bar.fill_bar_progress_with("O");
            bar.fill_bar_remainder_with(" ");
            bar.set_status_text("Formatting Output");
            bar.set_total(static_cast<float>(nUserTraces));
        }
        for (int64_t uIdx = 0; uIdx < nUserTraces; ++uIdx) {
            const auto& i = mObj.relevantTraces[userTraceIndices[uIdx]];
            traces.emplace_back(i.deviceLabel.value());
            traces.back().fileIndex = i.fIndex;
            if (i.storageType == StorageType::Voltage) {
                traces.back().type_ = 'V';
            } else if (i.storageType == StorageType::Phase) {
                traces.back().type_ = 'P';
            } else {
                traces.back().type_ = 'I';
            }
            traces.back().data_.reserve(result_indices.size());
        }
        auto process_trace = [&](int64_t uIdx) {
            const auto& i        = mObj.relevantTraces[userTraceIndices[uIdx]];
            auto&       outTrace = traces[uIdx + 1];
            auto&       st       = i.storageType;
            if (!iObj.argMin && uIdx % 4 == 0) { bar.update(static_cast<float>(uIdx)); }
            // Shorthand for the optional value
            auto                       i1    = i.index1.has_value() ? i.index1.value() : -1;
            auto                       i2    = i.index2.has_value() ? i.index2.value() : -1;
            auto                       vi    = i.variableIndex.has_value() ? i.variableIndex.value() : -1;
            const std::vector<double>* v1    = (i1 != -1) ? &(*x[i1]) : nullptr;
            const std::vector<double>* v2    = (i2 != -1) ? &(*x[i2]) : nullptr;
            const std::vector<double>* vv    = (vi != -1) ? &(*x[vi]) : nullptr;
            // Size of original data vectors available to FIR filter
            auto                       xsize = t.size();
            if (v1 && v1->size() < xsize) { xsize = v1->size(); }
            if (v2 && v2->size() < xsize) { xsize = v2->size(); }
            if (vv && vv->size() < xsize) { xsize = vv->size(); }
            // If this is a voltage we are storing
            if (st == StorageType::Voltage) {
                // Push the FIR filtered values for each value on the time axis
                for (auto j : result_indices) { // j: Index of window center
                    // FIR filter voltage
                    double accumulator = 0.;
                    for (int64_t k = 0; k < firsize; k++) { // k: Index used for the window
                        int64_t jk = j + halffirsize - k;   // jk: Index used for the data
                        // Sample only valid indices
                        if (jk < 0) {
                            jk = 0;
                        } else if (static_cast<size_t>(jk) >= xsize) {
                            jk = xsize - 1;
                        }
                        // Calculate individual voltage sample
                        double value;
                        // Shorthand values
                        auto   valin1 = v1 ? (*v1)[jk] : 0.0;
                        auto   valin2 = v2 ? (*v2)[jk] : 0.0;
                        // If the analysis method was voltage
                        if (iObj.argAnal == AnalysisType::Voltage) {
                            value = valin1 - valin2;
                            // Else calculate the voltage from the phase value
                        } else {
                            if (i.deviceLabel.value().at(3) == 'B' && vv) {
                                value = (*vv)[jk];
                            } else {
                                double valin1n1, valin1n2;
                                double valin2n1, valin2n2;
                                if (jk >= 2) {
                                    valin1n2 = v1 ? (*v1)[jk - 2] : 0.0;
                                    valin1n1 = v1 ? (*v1)[jk - 1] : 0.0;
                                    valin2n2 = v2 ? (*v2)[jk - 2] : 0.0;
                                    valin2n1 = v2 ? (*v2)[jk - 1] : 0.0;
                                } else if (jk == 1) {
                                    valin1n2 = valin1n1 = v1 ? (*v1)[jk - 1] : 0.0;
                                    valin2n2 = valin2n1 = v2 ? (*v2)[jk - 1] : 0.0;
                                } else {
                                    valin1n2 = valin1n1 = valin1;
                                    valin2n2 = valin2n1 = valin2;
                                }
                                value = ((3.0 * Constants::SIGMA) / (2.0 * iObj.transSim.tstep()))
                                        * ((valin1 - valin2) - (4.0 / 3.0) * (valin1n1 - valin2n1)
                                           + (1.0 / 3.0) * (valin1n2 - valin2n2));
                            }
                        }
                        accumulator += firwindow[k] * value;
                    }
                    outTrace.data_.emplace_back(accumulator);
                }
            } else if (st == StorageType::Phase) {
                // If the analysis type is phase
                if (iObj.argAnal == AnalysisType::Phase) {
                    // Push the FIR filtered values for each value on the time axis
                    for (auto j : result_indices) { // j: Index of window center
                        // FIR filter phase
                        double accumulator = 0.;
                        for (int64_t k = 0; k < firsize; k++) { // k: Index used for the window
                            int64_t jk = j + halffirsize - k;   // jk: Index used for the data
                            // Sample only valid indices
                            if (jk < 0) {
                                jk = 0;
                            } else if (static_cast<size_t>(jk) >= xsize) {
                                jk = xsize - 1;
                            }
                            // Calculate individual phase sample
                            auto valin1  = v1 ? (*v1)[jk] : 0.0;
                            auto valin2  = v2 ? (*v2)[jk] : 0.0;
                            auto value   = valin1 - valin2;
                            accumulator += firwindow[k] * value;
                        }
                        outTrace.data_.emplace_back(accumulator);
                    }
                    // Else if the analysis type was voltage
                } else {
                    if (i.deviceLabel.value().at(3) == 'B' && vv) {
                        // Push the FIR filtered values for each value on the time axis
                        for (auto j : result_indices) { // j: Index of window center
                            // FIR filter phase
                            double accumulator = 0.;
                            for (int64_t k = 0; k < firsize; k++) { // k: Index used for the window
                                int64_t jk = j + halffirsize - k;   // jk: Index used for the data
                                // Sample only valid indices
                                if (jk < 0) {
                                    jk = 0;
                                } else if (static_cast<size_t>(jk) >= xsize) {
                                    jk = xsize - 1;
                                }
                                // Get individual phase sample
                                auto value   = (*vv)[jk];
                                accumulator += firwindow[k] * value;
                            }
                            outTrace.data_.emplace_back(accumulator);
                        }
                    } else {
                        // Calculate the phase for all points in time.
                        // This is required for proper integration in case of voltage analysis.
                        // It takes extra memory and time, so do it only if necessary.
                        std::vector<double> vectphase;
                        vectphase.reserve(xsize);
                        double phaseN1 = 0., phaseN2 = phaseN1;
                        for (size_t j = 0; j < xsize; j++) {
                            auto valin1 = v1 ? (*v1)[j] : 0.0;
                            auto valin2 = v2 ? (*v2)[j] : 0.0;
                            auto value  = ((2.0 * iObj.transSim.tstep()) / (3.0 * Constants::SIGMA)) * (valin1 - valin2)
                                         + (4.0 / 3.0) * (phaseN1) - (1.0 / 3.0) * (phaseN2);
                            phaseN2 = phaseN1;
                            phaseN1 = value;
                            vectphase.emplace_back(value);
                        }
                        // Push the FIR filtered values for each value on the time axis
                        for (auto j : result_indices) { // j: Index of window center
                            // FIR filter phase
                            double accumulator = 0.;
                            for (int64_t k = 0; k < firsize; k++) { // k: Index used for the window
                                int64_t jk = j + halffirsize - k;   // jk: Index used for the data
                                // Sample only valid indices
                                if (jk < 0) {
                                    jk = 0;
                                } else if (static_cast<size_t>(jk) >= xsize) {
                                    jk = xsize - 1;
                                }
                                // Get individual phase sample
                                auto value   = vectphase[jk];
                                accumulator += firwindow[k] * value;
                            }
                            outTrace.data_.emplace_back(accumulator);
                        }
                    }
                }
            } else if (st == StorageType::Current) {
                Function localSrc;
                bool     isISrc = (i.deviceLabel.value().at(3) == 'I');
                if (isISrc) { localSrc = mObj.sourcegen.at(i.sourceIndex.value()); }
                // Push the FIR filtered values for each value on the time axis
                for (auto j : result_indices) { // j: Index of window center
                    // FIR filter current
                    double accumulator = 0.;
                    for (int64_t k = 0; k < firsize; k++) { // k: Index used for the window
                        int64_t jk = j + halffirsize - k;   // jk: Index used for the data
                        // Sample only valid indices
                        // Also apply index clipping to sourcegen in order to filter everything the same.
                        if (jk < 0) {
                            jk = 0;
                        } else if (static_cast<size_t>(jk) >= xsize) {
                            jk = xsize - 1;
                        }
                        // Get individual current sample
                        double value;
                        if (!isISrc) {
                            value = (*v1)[jk];
                        } else {
                            value = localSrc.value(t[jk]);
                        }
                        accumulator += firwindow[k] * value;
                    }
                    outTrace.data_.emplace_back(accumulator);
                }
            }
        };
#ifdef _OPENMP
        if (iObj.argPar && nUserTraces >= 2) {
            const int nThreads = std::min<int>(omp_get_max_threads(), static_cast<int>(nUserTraces));
#pragma omp parallel for schedule(dynamic) num_threads(nThreads)
            for (int64_t uIdx = 0; uIdx < nUserTraces; ++uIdx) { process_trace(uIdx); }
        } else
#endif
        {
            for (int64_t uIdx = 0; uIdx < nUserTraces; ++uIdx) { process_trace(uIdx); }
        }
        if (!iObj.argMin) {
            bar.complete();
            std::cout << "\n";
        }
    } else {
        ProgressBar bar;
        if (!iObj.argMin) {
            bar.create_thread();
            bar.set_bar_width(30);
            bar.fill_bar_progress_with("O");
            bar.fill_bar_remainder_with(" ");
            bar.set_status_text("Formatting Output");
            bar.set_total((float) mObj.nm.size());
        }
        std::vector<int64_t> nodeIndices;
        nodeIndices.reserve(mObj.nm.size());
        for (const auto& i : mObj.nm) {
            if (iObj.argAnal == AnalysisType::Voltage) {
                traces.emplace_back("V(" + i.first + ")");
                traces.back().type_ = 'V';
            } else {
                traces.emplace_back("P(" + i.first + ")");
                traces.back().type_ = 'P';
            }
            traces.back().data_.reserve(result_indices.size());
            nodeIndices.push_back(i.second);
        }
        const int64_t nNodes      = static_cast<int64_t>(nodeIndices.size());
        auto          process_node = [&](int64_t idx) {
            if (!iObj.argMin && idx % 64 == 0) { bar.update(static_cast<float>(idx)); }
            auto&                      outTrace = traces[idx + 1];
            const std::vector<double>& xv       = *x[nodeIndices[idx]];
            const size_t               xsize    = xv.size();
            const double*              xvPtr    = xv.data();
            for (auto j : result_indices) {
                double accumulator = 0.;
                for (int64_t k = 0; k < firsize; k++) {
                    int64_t jk = j + halffirsize - k;
                    if (jk < 0) {
                        jk = 0;
                    } else if (static_cast<size_t>(jk) >= xsize) {
                        jk = xsize - 1;
                    }
                    accumulator += firwindow[k] * xvPtr[jk];
                }
                outTrace.data_.emplace_back(accumulator);
            }
        };
#ifdef _OPENMP
        if (iObj.argPar && nNodes >= 2) {
            const int nThreads = std::min<int>(omp_get_max_threads(), static_cast<int>(nNodes));
#pragma omp parallel for schedule(static) num_threads(nThreads)
            for (int64_t idx = 0; idx < nNodes; ++idx) { process_node(idx); }
        } else
#endif
        {
            for (int64_t idx = 0; idx < nNodes; ++idx) { process_node(idx); }
        }
        if (!iObj.argMin) {
            bar.update(100);
            bar.complete();
            std::cout << "\n";
        }
    }
}

void Output::format_csv_or_dat(const std::string& filename, const char& delimiter, bool argmin, int64_t fIndex) {
    std::vector<int64_t> tIndices = {0};
    for (size_t i = 1; i < traces.size(); ++i) {
        if (traces[i].fileIndex == fIndex || fIndex == -1) { tIndices.emplace_back(i); }
    }
    std::ofstream     outfile(filename);
    std::vector<char> ioBuf(262144);
    outfile.rdbuf()->pubsetbuf(ioBuf.data(), static_cast<std::streamsize>(ioBuf.size()));
    outfile << std::setprecision(15);
    if (outfile.is_open()) {
        for (size_t i = 0; i < tIndices.size() - 1; ++i) { outfile << traces[tIndices[i]].name_ << delimiter; }
        outfile << traces[tIndices.back()].name_ << "\n";
        ProgressBar bar;
        const size_t numRows = traces[0].data_.size();
        const size_t numCols = tIndices.size();
        if (!argmin) {
            bar.create_thread();
            bar.set_bar_width(30);
            bar.fill_bar_progress_with("O");
            bar.fill_bar_remainder_with(" ");
            bar.set_status_text("Writing Output");
            bar.set_total((float) numRows);
        }
        outfile << std::scientific << std::setprecision(6);
        std::vector<const double*> colPtrs(numCols);
        for (size_t i = 0; i < numCols; ++i) { colPtrs[i] = traces[tIndices[i]].data_ptr(); }
        for (size_t j = 0; j < numRows; ++j) {
            if (!argmin) { bar.update(static_cast<float>(j)); }
            for (size_t i = 0; i < numCols - 1; ++i) {
                outfile << colPtrs[i][j] << delimiter;
            }
            outfile << colPtrs[numCols - 1][j] << "\n";
        }
        if (!argmin) {
            bar.complete();
            std::cout << "\n\n";
        }
    } else {
        Errors::output_errors(OutputErrors::CANNOT_OPEN_FILE, filename);
    }
}

void Output::format_bin(const std::string& filename, bool argmin, int64_t fIndex) {
    std::vector<int64_t> tIndices = {0};
    for (size_t i = 1; i < traces.size(); ++i) {
        if (traces[i].fileIndex == fIndex || fIndex == -1) { tIndices.emplace_back(i); }
    }
    std::ofstream outfile(filename, std::ios::out | std::ios::binary);
    if (!outfile.is_open()) {
        Errors::output_errors(OutputErrors::CANNOT_OPEN_FILE, filename);
        return;
    }
    if (traces.empty()) {
        Errors::output_errors(OutputErrors::NOTHING_SPECIFIED);
        return;
    }
    const uint64_t numCols   = static_cast<uint64_t>(tIndices.size());
    const uint64_t numRows   = static_cast<uint64_t>(traces[0].data_.size());
    const uint32_t version   = 1;
    const uint32_t flags     = 0;
    const char     magic[8]  = {'J', 'O', 'S', 'I', 'M', 'B', 'I', 'N'};

    outfile.write(magic, 8);
    outfile.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));
    outfile.write(reinterpret_cast<const char*>(&flags), sizeof(uint32_t));
    outfile.write(reinterpret_cast<const char*>(&numCols), sizeof(uint64_t));
    outfile.write(reinterpret_cast<const char*>(&numRows), sizeof(uint64_t));

    for (size_t c = 0; c < numCols; ++c) {
        const auto&    tr      = traces[tIndices[c]];
        const uint32_t nameLen = static_cast<uint32_t>(tr.name_.size());
        outfile.write(reinterpret_cast<const char*>(&nameLen), sizeof(uint32_t));
        outfile.write(tr.name_.data(), nameLen);
        outfile.write(&tr.type_, 1);
    }

    ProgressBar bar;
    if (!argmin) {
        bar.create_thread();
        bar.set_bar_width(30);
        bar.fill_bar_progress_with("O");
        bar.fill_bar_remainder_with(" ");
        bar.set_status_text("Writing Binary Output");
        bar.set_total(static_cast<float>(numCols));
    }
    const std::streamsize colBytes = static_cast<std::streamsize>(numRows * sizeof(double));
    for (size_t c = 0; c < numCols; ++c) {
        if (!argmin) { bar.update(static_cast<float>(c)); }
        outfile.write(reinterpret_cast<const char*>(traces[tIndices[c]].data_ptr()), colBytes);
    }
    if (!argmin) {
        bar.complete();
        std::cout << "\n\n";
    }
}

void Output::format_npy(const std::string& filename, bool argmin, int64_t fIndex) {
    std::vector<int64_t> tIndices = {0};
    for (size_t i = 1; i < traces.size(); ++i) {
        if (traces[i].fileIndex == fIndex || fIndex == -1) { tIndices.emplace_back(i); }
    }
    std::ofstream outfile(filename, std::ios::out | std::ios::binary);
    if (!outfile.is_open()) {
        Errors::output_errors(OutputErrors::CANNOT_OPEN_FILE, filename);
        return;
    }
    if (traces.empty()) {
        Errors::output_errors(OutputErrors::NOTHING_SPECIFIED);
        return;
    }
    const size_t numCols = tIndices.size();
    const size_t numRows = traces[0].data_.size();

    std::string dict = "{'descr': [";
    for (size_t c = 0; c < numCols; ++c) {
        std::string name = traces[tIndices[c]].name_;
        if (name.size() >= 2 && name.front() == '"' && name.back() == '"') {
            name = name.substr(1, name.size() - 2);
        }
        std::string escaped;
        escaped.reserve(name.size());
        for (char ch : name) {
            if (ch == '\'' || ch == '\\') { escaped.push_back('\\'); }
            escaped.push_back(ch);
        }
        dict += "('" + escaped + "', '<f8')";
        if (c + 1 < numCols) { dict += ", "; }
    }
    dict += "], 'fortran_order': False, 'shape': (" + std::to_string(numRows) + ",), }";

    const bool   useV2     = (10 + dict.size() + 1 > 65535);
    const size_t prefixLen = useV2 ? 12 : 10;
    const size_t rem       = (prefixLen + dict.size() + 1) % 64;
    const size_t padLen    = (rem == 0) ? 0 : (64 - rem);
    dict.append(padLen, ' ');
    dict.push_back('\n');

    const char magic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    outfile.write(magic, 6);
    if (!useV2) {
        const char     ver[2]  = {'\x01', '\x00'};
        const uint16_t hdrSize = static_cast<uint16_t>(dict.size());
        outfile.write(ver, 2);
        outfile.write(reinterpret_cast<const char*>(&hdrSize), sizeof(uint16_t));
    } else {
        const char     ver[2]  = {'\x02', '\x00'};
        const uint32_t hdrSize = static_cast<uint32_t>(dict.size());
        outfile.write(ver, 2);
        outfile.write(reinterpret_cast<const char*>(&hdrSize), sizeof(uint32_t));
    }
    outfile.write(dict.data(), static_cast<std::streamsize>(dict.size()));

    std::vector<const double*> colPtrs(numCols);
    for (size_t c = 0; c < numCols; ++c) { colPtrs[c] = traces[tIndices[c]].data_ptr(); }

    constexpr size_t    R_TILE       = 64;
    constexpr size_t    C_TILE       = 8;
    const size_t        rawRows      = std::max<size_t>(R_TILE, (2 * 1024 * 1024) / std::max<size_t>(1, numCols));
    const size_t        rowsPerChunk = std::min<size_t>(1024, ((rawRows + R_TILE - 1) / R_TILE) * R_TILE);
    std::vector<double> chunkBuf(rowsPerChunk * numCols);
    double              tile[R_TILE * C_TILE];
    for (size_t rowStart = 0; rowStart < numRows; rowStart += rowsPerChunk) {
        const size_t rowEnd = std::min(numRows, rowStart + rowsPerChunk);
        const size_t count  = rowEnd - rowStart;
        for (size_t r0 = 0; r0 < count; r0 += R_TILE) {
            const size_t rCnt = std::min(R_TILE, count - r0);
            for (size_t c0 = 0; c0 < numCols; c0 += C_TILE) {
                const size_t cCnt = std::min(C_TILE, numCols - c0);
                for (size_t dc = 0; dc < cCnt; ++dc) {
                    const double* src = colPtrs[c0 + dc] + rowStart + r0;
                    for (size_t r = 0; r < rCnt; ++r) { tile[r * C_TILE + dc] = src[r]; }
                }
                for (size_t r = 0; r < rCnt; ++r) {
                    std::copy_n(&tile[r * C_TILE], cCnt, &chunkBuf[(r0 + r) * numCols + c0]);
                }
            }
        }
        outfile.write(reinterpret_cast<const char*>(chunkBuf.data()),
                      static_cast<std::streamsize>(count * numCols * sizeof(double)));
    }
}

void Output::format_raw_bin(const std::string& filename, bool argmin, int64_t fIndex) {
    std::vector<int64_t> tIndices = {0};
    for (size_t i = 1; i < traces.size(); ++i) {
        if (traces[i].fileIndex == fIndex || fIndex == -1) { tIndices.emplace_back(i); }
    }
    std::ofstream outfile(filename, std::ios::out | std::ios::binary);
    if (!outfile.is_open()) {
        Errors::output_errors(OutputErrors::CANNOT_OPEN_FILE, filename);
        return;
    }
    if (traces.empty()) {
        Errors::output_errors(OutputErrors::NOTHING_SPECIFIED);
        return;
    }
    const size_t numCols = tIndices.size();
    const size_t numRows = traces[0].data_.size();

    outfile << "Title: " << std::filesystem::path(filename).stem().string() << "\n";
    std::time_t result = std::time(nullptr);
    outfile << "Date: " << std::asctime(std::localtime(&result));
    outfile << "Plotname: Transient Analysis\n";
    outfile << "Flags: real\n";
    outfile << "No. Variables: " << numCols << "\n";
    outfile << "No. Points: " << numRows << "\n";
    outfile << "Command: JoSIM v" << VERSION << "\n";
    outfile << "Variables:\n";
    outfile << " 0 time Seconds\n";
    for (size_t i = 1; i < numCols; ++i) {
        std::string name = traces[tIndices[i]].name_;
        name.erase(std::remove(name.begin(), name.end(), '\"'), name.end());
        const char t = traces[tIndices[i]].type_;
        if (t == 'V') {
            outfile << " " << i << " " << name << " Voltage\n";
        } else if (t == 'P') {
            outfile << " " << i << " " << name << " Phase\n";
        } else if (t == 'I') {
            std::replace(name.begin(), name.end(), '|', '.');
            if (name.size() > 3) { name = name.substr(2, name.size() - 3); }
            outfile << " " << i << " " << name << "#branch Current\n";
        }
    }
    outfile << "Binary:\n";

    std::vector<const double*> colPtrs(numCols);
    for (size_t c = 0; c < numCols; ++c) { colPtrs[c] = traces[tIndices[c]].data_ptr(); }

    constexpr size_t    R_TILE       = 64;
    constexpr size_t    C_TILE       = 8;
    const size_t        rawRows      = std::max<size_t>(R_TILE, (2 * 1024 * 1024) / std::max<size_t>(1, numCols));
    const size_t        rowsPerChunk = std::min<size_t>(1024, ((rawRows + R_TILE - 1) / R_TILE) * R_TILE);
    std::vector<double> chunkBuf(rowsPerChunk * numCols);
    double              tile[R_TILE * C_TILE];
    for (size_t rowStart = 0; rowStart < numRows; rowStart += rowsPerChunk) {
        const size_t rowEnd = std::min(numRows, rowStart + rowsPerChunk);
        const size_t count  = rowEnd - rowStart;
        for (size_t r0 = 0; r0 < count; r0 += R_TILE) {
            const size_t rCnt = std::min(R_TILE, count - r0);
            for (size_t c0 = 0; c0 < numCols; c0 += C_TILE) {
                const size_t cCnt = std::min(C_TILE, numCols - c0);
                for (size_t dc = 0; dc < cCnt; ++dc) {
                    const double* src = colPtrs[c0 + dc] + rowStart + r0;
                    for (size_t r = 0; r < rCnt; ++r) { tile[r * C_TILE + dc] = src[r]; }
                }
                for (size_t r = 0; r < rCnt; ++r) {
                    std::copy_n(&tile[r * C_TILE], cCnt, &chunkBuf[(r0 + r) * numCols + c0]);
                }
            }
        }
        outfile.write(reinterpret_cast<const char*>(chunkBuf.data()),
                      static_cast<std::streamsize>(count * numCols * sizeof(double)));
    }
}

// Writes the output to a standard spice raw file
void Output::format_raw(const std::string& filename, bool argmin, int64_t fIndex) {
    std::vector<int64_t> tIndices = {0};
    for (auto i = 1; i < traces.size(); ++i) {
        if (traces.at(i).fileIndex == fIndex || fIndex == -1) { tIndices.emplace_back(i); }
    }
    // Variable to store the total number of points to be saved
    int64_t       loopsize = 0;
    // Opens an output stream with provided file name
    std::ofstream outfile(filename);
    // Set the output presicion
    outfile << std::setprecision(6);
    // Ensure that the file could be opened
    if (outfile.is_open()) {
        // If the traces are not empty
        if (!traces.empty()) {
            // Write the rawfile preamble
            outfile << "Title: " << std::filesystem::path(filename).stem() << "\n";
            std::time_t result = std::time(nullptr);
            outfile << "Date: " << std::asctime(std::localtime(&result));
            outfile << "Plotname: Transient Analysis\n";
            outfile << "Flags: real\n";
            outfile << "No. Variables: " << tIndices.size() << "\n";
            loopsize = traces.at(0).data_.size();
            outfile << "No. Points: " << loopsize << "\n";
            outfile << "Command: JoSIM v" << VERSION << "\n";
            outfile << "Variables:\n";
            outfile << " 0 time Seconds\n";
            // Determine the variable name and type in a format acceptable for raw
            for (auto i = 1; i < tIndices.size(); ++i) {
                // Make a copy of the name so we can alter it
                std::string name = traces.at(tIndices.at(i)).name_;
                // Erase all the double quote characters
                name.erase(std::remove(name.begin(), name.end(), '\"'), name.end());
                // If this is a voltage variable
                if (traces.at(tIndices.at(i)).type_ == 'V') {
                    outfile << " " << i << " " << name << " Voltage\n";
                    // If this is a phase variable
                } else if (traces.at(tIndices.at(i)).type_ == 'P') {
                    outfile << " " << i << " " << name << " Phase\n";
                    // If this is a current variable
                } else if (traces.at(tIndices.at(i)).type_ == 'I') {
                    std::replace(name.begin(), name.end(), '|', '.');
                    name = name.substr(2);
                    name = name.substr(0, name.size() - 1);
                    // Append '#branch' since currents always flow in branches
                    outfile << " " << i << " " << name << "#branch Current\n";
                }
            }
            // Start filling the values
            outfile << "Values:\n";
            int64_t     pointSizeSpacing = std::to_string(loopsize).length() + 1;
            ProgressBar bar;
            if (!argmin) {
                bar.create_thread();
                bar.set_bar_width(30);
                bar.fill_bar_progress_with("O");
                bar.fill_bar_remainder_with(" ");
                bar.set_status_text("Writing Output");
                bar.set_total((float) loopsize);
            }
            for (int64_t i = 0; i < loopsize; ++i) {
                if (!argmin) { bar.update(static_cast<float>(i)); }
                // Point and time value
                outfile << std::left << std::setw(pointSizeSpacing) << i << traces.at(0).data_.at(i) << "\n";
                // Fill in rest of variable values
                for (auto j = 1; j < tIndices.size(); ++j) {
                    outfile << std::left << std::setw(pointSizeSpacing) << ""
                            << traces.at(tIndices.at(j)).data_ptr()[i] << "\n";
                }
            }
            if (!argmin) {
                bar.complete();
                std::cout << "\n\n";
            }
            // If the traces were empty (aka nothing to plot)
        } else if (traces.empty()) {
            // Complain about it
            Errors::output_errors(OutputErrors::NOTHING_SPECIFIED);
        }
        outfile.close();
        // If the file could not be opened for writing
    } else {
        // Complain about it
        Errors::output_errors(OutputErrors::CANNOT_OPEN_FILE, filename);
    }
}

void Output::format_cout(const bool& argMin) {
    if (!argMin) {
        for (auto i = 0; i < traces.size() - 1; ++i) { std::cout << traces.at(i).name_ << " "; }
        std::cout << traces.at(traces.size() - 1).name_ << "\n";
        for (auto j = 0; j < traces.at(0).data_.size(); ++j) {
            for (auto i = 0; i < traces.size() - 1; ++i) {
                std::cout << std::setw(15) << std::scientific << std::setprecision(6) << traces.at(i).data_ptr()[j]
                          << " ";
            }
            std::cout << std::setw(15) << std::scientific << std::setprecision(6)
                      << traces.at(traces.size() - 1).data_ptr()[j] << "\n";
        }
    }
}

void Output::write_events(const Input& iObj, Matrix& mObj, Simulation& sObj) {
    const auto& x    = sObj.results.xVector;
    const auto& t    = sObj.results.timeAxis;
    const auto& tran = iObj.transSim;

    std::vector<int64_t> result_indices;
    if (!sObj.results.printIndices.empty()) {
        result_indices = sObj.results.printIndices;
    } else if (!t.empty()) {
        for (size_t i = 0; i < t.size(); ++i) {
            if (t[i] >= tran.prstart()) {
                result_indices.emplace_back(static_cast<int64_t>(i));
                break;
            }
        }
        double next_print_point = tran.prstart() + tran.prstep();
        if (!result_indices.empty()) {
            for (size_t i = static_cast<size_t>(result_indices.back()); i < t.size(); ++i) {
                if (t[i] >= next_print_point) {
                    result_indices.emplace_back(static_cast<int64_t>(i));
                    next_print_point += tran.prstep();
                }
            }
        }
    }

    std::vector<RelevantTrace> fallbackTraces;
    const auto*                traceListPtr = &mObj.relevantTraces;
    if (mObj.relevantTraces.empty()) {
        for (const auto& devVar : mObj.components.devices) {
            if (std::holds_alternative<JJ>(devVar)) {
                const auto&   jjDev = std::get<JJ>(devVar);
                RelevantTrace temp;
                temp.device      = true;
                temp.storageType = StorageType::Phase;
                temp.deviceLabel = "\"P(" + jjDev.netlistInfo.label_ + ")\"";
                if (iObj.argAnal == AnalysisType::Voltage) {
                    temp.variableIndex = jjDev.variableIndex_;
                } else {
                    temp.index1 = jjDev.indexInfo.posIndex_;
                    temp.index2 = jjDev.indexInfo.negIndex_;
                }
                fallbackTraces.push_back(temp);
            }
        }
        traceListPtr = &fallbackTraces;
    }

    std::vector<std::string> traceNames;
    std::vector<SfqEvent>    allEvents;
    size_t                   maxNameLen = 1;

    const int64_t B = sObj.results.mcRuns;
    if (B > 1 && sObj.eventsOnly_ && !sObj.mcEventTrackers_.empty()) {
        for (size_t tIdx = 0; tIdx < traceListPtr->size(); ++tIdx) {
            const auto& tr = (*traceListPtr)[tIdx];
            if (tr.storageType != StorageType::Phase) { continue; }
            std::string baseName = tr.deviceLabel.value_or("P");
            baseName.erase(std::remove(baseName.begin(), baseName.end(), '\"'), baseName.end());
            for (int64_t b = 0; b < B; ++b) {
                std::string   name         = baseName + "_MC" + std::to_string(b);
                const int32_t phaseTraceId = static_cast<int32_t>(traceNames.size());
                traceNames.push_back(name);
                maxNameLen = std::max(maxNameLen, name.size());
                if (static_cast<size_t>(b) < sObj.mcEventTrackers_.size()
                    && tIdx < sObj.mcEventTrackers_[b].size()) {
                    const auto& evs = sObj.mcEventTrackers_[b][tIdx].events;
                    for (const auto& ev : evs) {
                        SfqEvent copyEv = ev;
                        copyEv.traceId  = phaseTraceId;
                        allEvents.push_back(copyEv);
                    }
                }
            }
        }
        if (iObj.argEventsOnly && !iObj.argMin && iObj.argSweep && !iObj.sweeps.empty()) {
            std::cout << "\nJoSIM Parameter Sweep / Margin Summary (" << B << " sweep points, events-only, tol = +/-"
                      << std::fixed << std::setprecision(1) << (iObj.argMarginTol * 1e12) << " ps):\n";
            for (size_t d = 0; d < iObj.sweeps.size(); ++d) {
                const auto& sd = iObj.sweeps[d];
                if (!sd.values.empty()) {
                    std::cout << "  Dim " << d << " (" << sd.target << "): " << sd.values.size() << " pts ["
                              << std::defaultfloat << std::setprecision(4) << sd.values.front() << " .. "
                              << sd.values.back() << "]" << (sd.isRelative ? " (relative scale)" : "") << "\n";
                }
            }
            const bool is1D = (iObj.sweeps.size() == 1 && static_cast<int64_t>(iObj.sweeps[0].values.size()) == B);
            int64_t    bNom = B / 2;
            auto       find_nom = [](const SweepDim& sd) -> int64_t {
                if (sd.values.empty()) { return 0; }
                if (sd.isRelative) {
                    int64_t best = 0;
                    for (size_t k = 1; k < sd.values.size(); ++k) {
                        if (std::abs(sd.values[k] - 1.0) < std::abs(sd.values[best] - 1.0)) {
                            best = static_cast<int64_t>(k);
                        }
                    }
                    return best;
                }
                return static_cast<int64_t>(sd.values.size() / 2);
            };
            if (iObj.sweeps.size() == 1) {
                bNom = std::clamp<int64_t>(find_nom(iObj.sweeps[0]), 0, B - 1);
            } else if (iObj.sweeps.size() >= 2) {
                const int64_t n1 = static_cast<int64_t>(iObj.sweeps[1].values.size());
                bNom = std::clamp<int64_t>(find_nom(iObj.sweeps[0]) * n1 + find_nom(iObj.sweeps[1]), 0, B - 1);
            }
            auto seq_matches = [&](const std::vector<SfqEvent>& refEvs, const std::vector<SfqEvent>& curEvs) -> bool {
                const double tStop  = iObj.transSim.tstop();
                const double tolSec = iObj.argMarginTol;
                const size_t nCom   = std::min(refEvs.size(), curEvs.size());
                double       prevDt = 0.0;
                for (size_t p = 0; p < nCom; ++p) {
                    if (curEvs[p].polarity != refEvs[p].polarity) { return false; }
                    const double dt = curEvs[p].time - refEvs[p].time;
                    if (tolSec > 0.0 && std::abs(dt) > tolSec) { return false; }
                    prevDt = dt;
                }
                if (curEvs.size() == refEvs.size()) { return true; }
                if (tStop > 0.0 && refEvs.size() == nCom + 1 && nCom >= 1 && prevDt > 0.0) {
                    const double dispSlack = std::max(2.0e-12, 0.25 * prevDt);
                    if (refEvs[nCom].time + prevDt + dispSlack >= tStop) { return true; }
                }
                if (tStop > 0.0 && curEvs.size() == nCom + 1 && nCom >= 1 && prevDt < 0.0
                    && curEvs[nCom].polarity == refEvs[nCom - 1].polarity) {
                    const double dispSlack = std::max(2.0e-12, 0.25 * (-prevDt));
                    bool         spacingOk = true;
                    if (nCom >= 2) {
                        const double refPeriod = refEvs[nCom - 1].time - refEvs[nCom - 2].time;
                        const double curPeriod = curEvs[nCom].time - curEvs[nCom - 1].time;
                        if (refPeriod > 0.0 && curPeriod < 0.5 * refPeriod) { spacingOk = false; }
                    }
                    if (spacingOk && curEvs[nCom].time - prevDt + dispSlack >= tStop) { return true; }
                }
                return false;
            };
            std::cout << "  " << std::left << std::setw(22) << "Trace" << std::right << std::setw(10) << "Nom SFQ"
                      << std::setw(12) << "Pass Pts" << (is1D ? "  Operating Window" : "") << "\n";
            int64_t              shown = 0;
            std::vector<uint8_t> circuitPass(static_cast<size_t>(B), 1);
            int64_t              totalNomPulses = 0;
            for (size_t tIdx = 0; tIdx < traceListPtr->size(); ++tIdx) {
                const auto& tr = (*traceListPtr)[tIdx];
                if (tr.storageType != StorageType::Phase) { continue; }
                const auto&          refEvs = sObj.mcEventTrackers_[bNom][tIdx].events;
                const int64_t        refSfq = static_cast<int64_t>(refEvs.size());
                totalNomPulses             += refSfq;
                std::vector<uint8_t> trPass(static_cast<size_t>(B), 0);
                int64_t              passCnt = 0;
                for (int64_t b = 0; b < B; ++b) {
                    const auto& curEvs = sObj.mcEventTrackers_[b][tIdx].events;
                    if (seq_matches(refEvs, curEvs)) {
                        trPass[b] = 1;
                        ++passCnt;
                    } else {
                        circuitPass[b] = 0;
                    }
                }
                if (shown < 16) {
                    ++shown;
                    int64_t lo = bNom, hi = bNom;
                    while (lo > 0 && trPass[lo - 1]) { --lo; }
                    while (hi + 1 < B && trPass[hi + 1]) { ++hi; }
                    std::ostringstream passStr;
                    passStr << passCnt << "/" << B;
                    std::cout << "  " << std::left << std::setw(22) << tr.deviceLabel.value() << std::right
                              << std::setw(10) << refSfq << std::setw(12) << passStr.str();
                    if (is1D) {
                        const auto&  vals = iObj.sweeps[0].values;
                        const double vLo  = vals[lo];
                        const double vHi  = vals[hi];
                        std::cout << "  [" << std::defaultfloat << std::setprecision(4) << vLo << ", " << vHi << "]";
                        if (iObj.sweeps[0].isRelative) {
                            std::cout << " (" << std::showpos << std::fixed << std::setprecision(1)
                                      << (vLo - 1.0) * 100.0 << "%, " << (vHi - 1.0) * 100.0 << "%" << std::noshowpos
                                      << ")";
                        }
                    }
                    std::cout << "\n";
                }
            }
            if (shown > 0) {
                int64_t cPassCnt = 0;
                for (int64_t b = 0; b < B; ++b) {
                    if (circuitPass[b]) { ++cPassCnt; }
                }
                int64_t cLo = bNom, cHi = bNom;
                while (cLo > 0 && circuitPass[cLo - 1]) { --cLo; }
                while (cHi + 1 < B && circuitPass[cHi + 1]) { ++cHi; }
                std::ostringstream passStr;
                passStr << cPassCnt << "/" << B;
                std::cout << "  " << std::left << std::setw(22) << "[ALL_JJS_EXACT]" << std::right << std::setw(10)
                          << totalNomPulses << std::setw(12) << passStr.str();
                if (is1D) {
                    const auto&  vals = iObj.sweeps[0].values;
                    const double vLo  = vals[cLo];
                    const double vHi  = vals[cHi];
                    std::cout << "  [" << std::defaultfloat << std::setprecision(4) << vLo << ", " << vHi << "]";
                    if (iObj.sweeps[0].isRelative) {
                        std::cout << " (" << std::showpos << std::fixed << std::setprecision(1) << (vLo - 1.0) * 100.0
                                  << "%, " << (vHi - 1.0) * 100.0 << "%" << std::noshowpos << ")";
                    }
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    } else if (B > 1 && !sObj.results.mcTraceOut.empty() && !result_indices.empty()) {
        const int64_t nRows = sObj.results.mcNumRows;
        const int64_t nKept = sObj.results.mcNumKeptSteps;
        for (size_t tIdx = 0; tIdx < traceListPtr->size(); ++tIdx) {
            const auto& tr = (*traceListPtr)[tIdx];
            if (tr.storageType != StorageType::Phase) { continue; }
            std::string baseName = tr.deviceLabel.value_or("P");
            baseName.erase(std::remove(baseName.begin(), baseName.end(), '\"'), baseName.end());
            const int32_t s1 = (tr.index1 && *tr.index1 >= 0
                                && *tr.index1 < static_cast<int64_t>(sObj.results.mcRowToSlot.size()))
                                       ? sObj.results.mcRowToSlot[*tr.index1]
                                       : -1;
            const int32_t s2 = (tr.index2 && *tr.index2 >= 0
                                && *tr.index2 < static_cast<int64_t>(sObj.results.mcRowToSlot.size()))
                                       ? sObj.results.mcRowToSlot[*tr.index2]
                                       : -1;
            const int32_t sv = (tr.variableIndex && *tr.variableIndex >= 0
                                && *tr.variableIndex < static_cast<int64_t>(sObj.results.mcRowToSlot.size()))
                                       ? sObj.results.mcRowToSlot[*tr.variableIndex]
                                       : -1;
            for (int64_t b = 0; b < B; ++b) {
                std::string   name         = baseName + "_MC" + std::to_string(b);
                const int32_t phaseTraceId = static_cast<int32_t>(traceNames.size());
                traceNames.push_back(name);
                maxNameLen = std::max(maxNameLen, name.size());
                const double* p1 = (s1 >= 0) ? (sObj.results.mcTraceOut.data() + (b * nRows + s1) * nKept) : nullptr;
                const double* p2 = (s2 >= 0) ? (sObj.results.mcTraceOut.data() + (b * nRows + s2) * nKept) : nullptr;
                const double* pv = (sv >= 0) ? (sObj.results.mcTraceOut.data() + (b * nRows + sv) * nKept) : nullptr;
                InLoopEventTracker trk;
                trk.traceId = phaseTraceId;
                trk.sign    = 1;
                trk.init_metrics(iObj.argEventMetrics);
                for (size_t r = 0; r < result_indices.size(); ++r) {
                    const int64_t j   = result_indices[r];
                    double        phi = 0.0;
                    if (iObj.argAnal == AnalysisType::Voltage && pv != nullptr) {
                        phi = pv[j];
                    } else {
                        phi = (p1 ? p1[j] : 0.0) - (p2 ? p2[j] : 0.0);
                    }
                    trk.step(t[j], phi, iObj.argEventMetrics);
                }
                trk.finalize(iObj.argEventMetrics);
                allEvents.insert(allEvents.end(), trk.events.begin(), trk.events.end());
            }
        }
    } else {
        for (size_t tIdx = 0; tIdx < traceListPtr->size(); ++tIdx) {
            const auto& tr = (*traceListPtr)[tIdx];
            if (tr.storageType != StorageType::Phase) { continue; }
            std::string name = tr.deviceLabel.value_or("P");
            name.erase(std::remove(name.begin(), name.end(), '\"'), name.end());
            const int32_t phaseTraceId = static_cast<int32_t>(traceNames.size());
            traceNames.push_back(name);
            maxNameLen = std::max(maxNameLen, name.size());

            int64_t singleRow = -1;
            if (tr.variableIndex.has_value() && !tr.index1.has_value() && !tr.index2.has_value()) {
                singleRow = *tr.variableIndex;
            } else if (tr.index1.has_value() && !tr.index2.has_value() && !tr.variableIndex.has_value()) {
                singleRow = *tr.index1;
            } else if (!tr.index1.has_value() && tr.index2.has_value() && !tr.variableIndex.has_value()) {
                singleRow = *tr.index2;
            }

            if (sObj.eventsOnly_ && singleRow >= 0 && singleRow < static_cast<int64_t>(sObj.useEventTracker_.size())
                && sObj.useEventTracker_[singleRow]) {
                const auto& evs = sObj.eventTrackers_[singleRow].events;
                for (const auto& ev : evs) {
                    SfqEvent copyEv = ev;
                    copyEv.traceId  = phaseTraceId;
                    allEvents.push_back(copyEv);
                }
                continue;
            }

            const double* p1 = (tr.index1 && *tr.index1 >= 0 && *tr.index1 < static_cast<int64_t>(x.size()) && x[*tr.index1])
                                       ? x[*tr.index1]->data()
                                       : nullptr;
            const double* p2 = (tr.index2 && *tr.index2 >= 0 && *tr.index2 < static_cast<int64_t>(x.size()) && x[*tr.index2])
                                       ? x[*tr.index2]->data()
                                       : nullptr;
            const double* pv = (tr.variableIndex && *tr.variableIndex >= 0
                                && *tr.variableIndex < static_cast<int64_t>(x.size()) && x[*tr.variableIndex])
                                       ? x[*tr.variableIndex]->data()
                                       : nullptr;

            if (result_indices.empty()) { continue; }
            InLoopEventTracker trk;
            trk.traceId = phaseTraceId;
            trk.sign    = 1;
            trk.init_metrics(iObj.argEventMetrics);
            for (size_t r = 0; r < result_indices.size(); ++r) {
                const int64_t j   = result_indices[r];
                double        phi = 0.0;
                if (iObj.argAnal == AnalysisType::Voltage && pv != nullptr) {
                    phi = pv[j];
                } else {
                    phi = (p1 ? p1[j] : 0.0) - (p2 ? p2[j] : 0.0);
                }
                trk.step(t[j], phi, iObj.argEventMetrics);
            }
            trk.finalize(iObj.argEventMetrics);
            allEvents.insert(allEvents.end(), trk.events.begin(), trk.events.end());
        }
    }

    std::stable_sort(allEvents.begin(), allEvents.end(), [](const SfqEvent& a, const SfqEvent& b) {
        if (a.time != b.time) { return a.time < b.time; }
        return a.traceId < b.traceId;
    });

    const uint32_t mMask     = iObj.argEventMetrics;
    const bool     hasTPeak  = (mMask & EVENT_METRIC_T_PEAK) != 0u;
    const bool     hasVPeak  = (mMask & EVENT_METRIC_V_PEAK) != 0u;
    const bool     hasFwhm   = (mMask & EVENT_METRIC_FWHM) != 0u;
    const bool     hasDtPi   = (mMask & EVENT_METRIC_DT_PI) != 0u;
    const bool     hasArea   = (mMask & EVENT_METRIC_AREA) != 0u;
    const size_t   extraCols = static_cast<size_t>(hasTPeak) + static_cast<size_t>(hasVPeak)
                             + static_cast<size_t>(hasFwhm) + static_cast<size_t>(hasDtPi)
                             + static_cast<size_t>(hasArea);

    std::string ext = std::filesystem::path(*iObj.eventsFile).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".npy") {
        std::ofstream outfile(*iObj.eventsFile, std::ios::out | std::ios::binary);
        if (!outfile.is_open()) { Errors::output_errors(OutputErrors::CANNOT_OPEN_FILE, *iObj.eventsFile); }

        std::string extraDescr;
        if (hasTPeak) { extraDescr += ", ('t_peak', '<f8')"; }
        if (hasVPeak) { extraDescr += ", ('v_peak', '<f8')"; }
        if (hasFwhm) { extraDescr += ", ('fwhm', '<f8')"; }
        if (hasDtPi) { extraDescr += ", ('dt_pi', '<f8')"; }
        if (hasArea) { extraDescr += ", ('area', '<f8')"; }

        std::string dict = "{'descr': [('trace_id', '<i4'), ('trace', '|S" + std::to_string(maxNameLen)
                           + "'), ('time', '<f8'), ('polarity', '<i4'), ('winding', '<i4'), ('phase', '<f8')"
                           + extraDescr + "], 'fortran_order': False, 'shape': ("
                           + std::to_string(allEvents.size()) + ",), }";
        const size_t preambleLen = 10;
        size_t       totalLen    = preambleLen + dict.size() + 1;
        size_t       pad         = (64 - (totalLen % 64)) % 64;
        dict.append(pad, ' ');
        dict.push_back('\n');

        const char     magic[]   = "\x93NUMPY";
        const uint8_t  major     = 1;
        const uint8_t  minor     = 0;
        const uint16_t headerLen = static_cast<uint16_t>(dict.size());
        outfile.write(magic, 6);
        outfile.write(reinterpret_cast<const char*>(&major), 1);
        outfile.write(reinterpret_cast<const char*>(&minor), 1);
        outfile.write(reinterpret_cast<const char*>(&headerLen), 2);
        outfile.write(dict.data(), static_cast<std::streamsize>(dict.size()));

        const size_t      recSize = 4 + maxNameLen + 8 + 4 + 4 + 8 + 8 * extraCols;
        std::vector<char> buf(allEvents.size() * recSize, 0);
        for (size_t idx = 0; idx < allEvents.size(); ++idx) {
            const auto&        ev   = allEvents[idx];
            char*              dst  = buf.data() + idx * recSize;
            const std::string& tStr = traceNames[ev.traceId];
            std::memcpy(dst, &ev.traceId, 4);
            std::memcpy(dst + 4, tStr.data(), std::min(maxNameLen, tStr.size()));
            std::memcpy(dst + 4 + maxNameLen, &ev.time, 8);
            std::memcpy(dst + 4 + maxNameLen + 8, &ev.polarity, 4);
            std::memcpy(dst + 4 + maxNameLen + 12, &ev.winding, 4);
            std::memcpy(dst + 4 + maxNameLen + 16, &ev.phase, 8);
            size_t off = 4 + maxNameLen + 24;
            if (hasTPeak) {
                std::memcpy(dst + off, &ev.tPeak, 8);
                off += 8;
            }
            if (hasVPeak) {
                std::memcpy(dst + off, &ev.vPeak, 8);
                off += 8;
            }
            if (hasFwhm) {
                std::memcpy(dst + off, &ev.fwhm, 8);
                off += 8;
            }
            if (hasDtPi) {
                std::memcpy(dst + off, &ev.dtPi, 8);
                off += 8;
            }
            if (hasArea) {
                std::memcpy(dst + off, &ev.area, 8);
                off += 8;
            }
        }
        if (!buf.empty()) {
            outfile.write(buf.data(), static_cast<std::streamsize>(buf.size()));
        }
    } else {
        std::ofstream outfile(*iObj.eventsFile);
        if (!outfile.is_open()) { Errors::output_errors(OutputErrors::CANNOT_OPEN_FILE, *iObj.eventsFile); }
        outfile << "trace_id,trace,time,polarity,winding,phase";
        if (hasTPeak) { outfile << ",t_peak"; }
        if (hasVPeak) { outfile << ",v_peak"; }
        if (hasFwhm) { outfile << ",fwhm"; }
        if (hasDtPi) { outfile << ",dt_pi"; }
        if (hasArea) { outfile << ",area"; }
        outfile << "\n";
        outfile << std::scientific << std::setprecision(12);
        for (const auto& ev : allEvents) {
            outfile << ev.traceId << "," << traceNames[ev.traceId] << "," << ev.time << "," << ev.polarity << ","
                    << ev.winding << "," << ev.phase;
            if (hasTPeak) { outfile << "," << ev.tPeak; }
            if (hasVPeak) { outfile << "," << ev.vPeak; }
            if (hasFwhm) { outfile << "," << ev.fwhm; }
            if (hasDtPi) { outfile << "," << ev.dtPi; }
            if (hasArea) { outfile << "," << ev.area; }
            outfile << "\n";
        }
    }
}

namespace {
struct PhaseTraceInfo {
    size_t      relTraceIdx = 0;
    std::string label;
};

struct EventSeqDiagnosis {
    bool        passed        = true;
    bool        isStructural  = false;
    std::string kind          = "PASS"; // PASS, MISSING_PULSE, EXTRA_PULSE, POLARITY_ERR, TIMING_LATE, TIMING_EARLY
    std::string traceLabel;
    int32_t     pulseIndex    = -1; // 1-based pulse index on that trace
    double      failTime      = 0.0;
    double      refTime       = 0.0;
    double      actualTime    = 0.0;
    double      timeDelta     = 0.0;
    double      maxTimingSkew = 0.0;
    int64_t     totalPulses   = 0;
};

std::string normalize_event_trace_label(std::string s) {
    s.erase(std::remove(s.begin(), s.end(), '\"'), s.end());
    s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
    s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    return s;
}

std::string format_diagnosis_short(const EventSeqDiagnosis& d) {
    if (d.passed) { return "PASS"; }
    std::ostringstream oss;
    oss << d.kind << " " << d.traceLabel << " #" << d.pulseIndex;
    if (d.kind == "MISSING_PULSE") {
        oss << " (exp " << std::fixed << std::setprecision(1) << (d.refTime * 1e12) << "ps)";
    } else if (d.kind == "EXTRA_PULSE") {
        oss << " @ " << std::fixed << std::setprecision(1) << (d.actualTime * 1e12) << "ps";
    } else if (d.kind == "POLARITY_ERR") {
        oss << " @ " << std::fixed << std::setprecision(1) << (d.actualTime * 1e12) << "ps";
    } else if (d.kind == "TIMING_LATE" || d.kind == "TIMING_EARLY") {
        oss << " (" << std::showpos << std::fixed << std::setprecision(1) << (d.timeDelta * 1e12) << "ps"
            << std::noshowpos << " @ " << (d.actualTime * 1e12) << "ps)";
    }
    return oss.str();
}

EventSeqDiagnosis compare_event_sequences(const std::vector<PhaseTraceInfo>&         phaseTraces,
                                          const std::vector<std::vector<SfqEvent>>& refEvents,
                                          const std::vector<std::vector<SfqEvent>>& curEvents,
                                          double                                    tolSec,
                                          double                                    tStop = 0.0) {
    EventSeqDiagnosis res;
    EventSeqDiagnosis bestStruct;
    EventSeqDiagnosis bestTiming;
    bool              hasStruct = false;
    bool              hasTiming = false;

    auto record_struct = [&](const std::string& kind,
                             const std::string& label,
                             int32_t            pulseIdx,
                             double             failT,
                             double             refT,
                             double             actT,
                             double             dt) {
        if (!hasStruct || failT < bestStruct.failTime) {
            hasStruct               = true;
            bestStruct.passed       = false;
            bestStruct.isStructural = true;
            bestStruct.kind         = kind;
            bestStruct.traceLabel   = label;
            bestStruct.pulseIndex   = pulseIdx;
            bestStruct.failTime     = failT;
            bestStruct.refTime      = refT;
            bestStruct.actualTime   = actT;
            bestStruct.timeDelta    = dt;
        }
    };

    const size_t nTraces = std::min(phaseTraces.size(), std::min(refEvents.size(), curEvents.size()));
    for (size_t k = 0; k < nTraces; ++k) {
        const auto& refEvs  = refEvents[k];
        const auto& curEvs  = curEvents[k];
        res.totalPulses    += static_cast<int64_t>(curEvs.size());

        size_t i            = 0;
        size_t j            = 0;
        double prevDt       = 0.0;
        bool   brokeOnTrace = false;

        while (i < refEvs.size() && j < curEvs.size()) {
            const double expT = refEvs[i].time + prevDt;

            if (i + 1 < refEvs.size() && (curEvs.size() - j) < (refEvs.size() - i)) {
                const double expNextT = refEvs[i + 1].time + prevDt;
                if (curEvs[j].polarity == refEvs[i + 1].polarity
                    && std::abs(curEvs[j].time - expNextT) < std::abs(curEvs[j].time - expT)) {
                    record_struct("MISSING_PULSE",
                                  phaseTraces[k].label,
                                  static_cast<int32_t>(i + 1),
                                  refEvs[i].time,
                                  refEvs[i].time,
                                  0.0,
                                  0.0);
                    brokeOnTrace = true;
                    break;
                }
            }

            if (j + 1 < curEvs.size() && (curEvs.size() - j) > (refEvs.size() - i)) {
                if (curEvs[j + 1].polarity == refEvs[i].polarity
                    && std::abs(curEvs[j + 1].time - expT) < std::abs(curEvs[j].time - expT)) {
                    record_struct("EXTRA_PULSE",
                                  phaseTraces[k].label,
                                  static_cast<int32_t>(j + 1),
                                  curEvs[j].time,
                                  0.0,
                                  curEvs[j].time,
                                  0.0);
                    brokeOnTrace = true;
                    break;
                }
            }

            if (curEvs[j].polarity != refEvs[i].polarity) {
                const double tFail = std::min(curEvs[j].time, refEvs[i].time);
                record_struct("POLARITY_ERR",
                              phaseTraces[k].label,
                              static_cast<int32_t>(i + 1),
                              tFail,
                              refEvs[i].time,
                              curEvs[j].time,
                              curEvs[j].time - refEvs[i].time);
                brokeOnTrace = true;
                break;
            }

            const double dt    = curEvs[j].time - refEvs[i].time;
            const double absDt = std::abs(dt);
            prevDt             = dt;
            if (absDt > res.maxTimingSkew) { res.maxTimingSkew = absDt; }
            if (tolSec > 0.0 && absDt > tolSec) {
                const double tFail = refEvs[i].time;
                if (!hasTiming || tFail < bestTiming.failTime) {
                    hasTiming               = true;
                    bestTiming.passed       = false;
                    bestTiming.isStructural = false;
                    bestTiming.kind         = (dt > 0.0) ? "TIMING_LATE" : "TIMING_EARLY";
                    bestTiming.traceLabel   = phaseTraces[k].label;
                    bestTiming.pulseIndex   = static_cast<int32_t>(i + 1);
                    bestTiming.failTime     = tFail;
                    bestTiming.refTime      = refEvs[i].time;
                    bestTiming.actualTime   = curEvs[j].time;
                    bestTiming.timeDelta    = dt;
                }
            }
            ++i;
            ++j;
        }

        if (!brokeOnTrace) {
            if (i < refEvs.size()) {
                bool horizonShifted = false;
                if (tStop > 0.0 && i + 1 == refEvs.size() && i >= 1 && prevDt > 0.0) {
                    const double dispSlack = std::max(2.0e-12, 0.25 * prevDt);
                    if (refEvs[i].time + prevDt + dispSlack >= tStop) { horizonShifted = true; }
                }
                if (!horizonShifted) {
                    record_struct("MISSING_PULSE",
                                  phaseTraces[k].label,
                                  static_cast<int32_t>(i + 1),
                                  refEvs[i].time,
                                  refEvs[i].time,
                                  0.0,
                                  0.0);
                }
            } else if (j < curEvs.size()) {
                bool horizonShifted = false;
                if (tStop > 0.0 && j + 1 == curEvs.size() && j >= 1 && i >= 1 && prevDt < 0.0
                    && curEvs[j].polarity == refEvs[i - 1].polarity) {
                    const double dispSlack = std::max(2.0e-12, 0.25 * (-prevDt));
                    bool         spacingOk = true;
                    if (i >= 2) {
                        const double refPeriod = refEvs[i - 1].time - refEvs[i - 2].time;
                        const double curPeriod = curEvs[j].time - curEvs[j - 1].time;
                        if (refPeriod > 0.0 && curPeriod < 0.5 * refPeriod) { spacingOk = false; }
                    }
                    if (spacingOk && curEvs[j].time - prevDt + dispSlack >= tStop) { horizonShifted = true; }
                }
                if (!horizonShifted) {
                    record_struct("EXTRA_PULSE",
                                  phaseTraces[k].label,
                                  static_cast<int32_t>(j + 1),
                                  curEvs[j].time,
                                  0.0,
                                  curEvs[j].time,
                                  0.0);
                }
            }
        }
    }

    if (hasStruct) {
        bestStruct.maxTimingSkew = res.maxTimingSkew;
        bestStruct.totalPulses   = res.totalPulses;
        return bestStruct;
    }
    if (hasTiming) {
        bestTiming.maxTimingSkew = res.maxTimingSkew;
        bestTiming.totalPulses   = res.totalPulses;
        return bestTiming;
    }
    return res;
}

void extract_trial_phase_events(const Input&                                       iObj,
                                const Matrix&                                      mObj,
                                const Simulation&                                  sObj,
                                std::vector<PhaseTraceInfo>&                       outPhaseTraces,
                                std::vector<std::vector<std::vector<SfqEvent>>>&   outTrialEvents) {
    outPhaseTraces.clear();
    outTrialEvents.clear();

    const auto& x    = sObj.results.xVector;
    const auto& t    = sObj.results.timeAxis;
    const auto& tran = iObj.transSim;

    std::vector<int64_t> result_indices;
    if (!sObj.results.printIndices.empty()) {
        result_indices = sObj.results.printIndices;
    } else if (!t.empty()) {
        for (size_t i = 0; i < t.size(); ++i) {
            if (t[i] >= tran.prstart()) {
                result_indices.emplace_back(static_cast<int64_t>(i));
                break;
            }
        }
        double next_print_point = tran.prstart() + tran.prstep();
        if (!result_indices.empty()) {
            for (size_t i = static_cast<size_t>(result_indices.back()); i < t.size(); ++i) {
                if (t[i] >= next_print_point) {
                    result_indices.emplace_back(static_cast<int64_t>(i));
                    next_print_point += tran.prstep();
                }
            }
        }
    }

    std::vector<RelevantTrace> fallbackTraces;
    const auto*                traceListPtr = &mObj.relevantTraces;
    if (mObj.relevantTraces.empty()) {
        for (const auto& devVar : mObj.components.devices) {
            if (std::holds_alternative<JJ>(devVar)) {
                const auto&   jjDev = std::get<JJ>(devVar);
                RelevantTrace temp;
                temp.device      = true;
                temp.storageType = StorageType::Phase;
                temp.deviceLabel = "\"P(" + jjDev.netlistInfo.label_ + ")\"";
                if (iObj.argAnal == AnalysisType::Voltage) {
                    temp.variableIndex = jjDev.variableIndex_;
                } else {
                    temp.index1 = jjDev.indexInfo.posIndex_;
                    temp.index2 = jjDev.indexInfo.negIndex_;
                }
                fallbackTraces.push_back(temp);
            }
        }
        traceListPtr = &fallbackTraces;
    }

    for (size_t tIdx = 0; tIdx < traceListPtr->size(); ++tIdx) {
        const auto& tr = (*traceListPtr)[tIdx];
        if (tr.storageType != StorageType::Phase) { continue; }
        std::string lbl = tr.deviceLabel.value_or("P");
        lbl.erase(std::remove(lbl.begin(), lbl.end(), '\"'), lbl.end());
        outPhaseTraces.push_back({tIdx, lbl});
    }

    const int64_t B      = std::max<int64_t>(1, sObj.results.mcRuns);
    const size_t  nPhase = outPhaseTraces.size();
    outTrialEvents.assign(static_cast<size_t>(B), std::vector<std::vector<SfqEvent>>(nPhase));
    if (nPhase == 0) { return; }

    if (B > 1 && sObj.eventsOnly_ && !sObj.mcEventTrackers_.empty()) {
        const auto& mcTrks = sObj.mcEventTrackers_;
        for (int64_t b = 0; b < B && static_cast<size_t>(b) < mcTrks.size(); ++b) {
            for (size_t k = 0; k < nPhase; ++k) {
                const size_t tIdx = outPhaseTraces[k].relTraceIdx;
                if (tIdx < mcTrks[b].size()) {
                    outTrialEvents[b][k] = mcTrks[b][tIdx].events;
                }
            }
        }
    } else if (B > 1 && !sObj.results.mcTraceOut.empty() && !result_indices.empty()) {
        const int64_t nRows = sObj.results.mcNumRows;
        const int64_t nKept = sObj.results.mcNumKeptSteps;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (iObj.argPar || iObj.argGpu)
#endif
        for (int64_t b = 0; b < B; ++b) {
            for (size_t k = 0; k < nPhase; ++k) {
                const auto&   tr = (*traceListPtr)[outPhaseTraces[k].relTraceIdx];
                const int32_t s1 = (tr.index1 && *tr.index1 >= 0
                                    && *tr.index1 < static_cast<int64_t>(sObj.results.mcRowToSlot.size()))
                                           ? sObj.results.mcRowToSlot[*tr.index1]
                                           : -1;
                const int32_t s2 = (tr.index2 && *tr.index2 >= 0
                                    && *tr.index2 < static_cast<int64_t>(sObj.results.mcRowToSlot.size()))
                                           ? sObj.results.mcRowToSlot[*tr.index2]
                                           : -1;
                const int32_t sv = (tr.variableIndex && *tr.variableIndex >= 0
                                    && *tr.variableIndex < static_cast<int64_t>(sObj.results.mcRowToSlot.size()))
                                           ? sObj.results.mcRowToSlot[*tr.variableIndex]
                                           : -1;
                const double* p1 = (s1 >= 0) ? (sObj.results.mcTraceOut.data() + (b * nRows + s1) * nKept) : nullptr;
                const double* p2 = (s2 >= 0) ? (sObj.results.mcTraceOut.data() + (b * nRows + s2) * nKept) : nullptr;
                const double* pv = (sv >= 0) ? (sObj.results.mcTraceOut.data() + (b * nRows + sv) * nKept) : nullptr;
                InLoopEventTracker trk;
                trk.traceId = static_cast<int32_t>(k);
                trk.sign    = 1;
                trk.init_metrics(iObj.argEventMetrics);
                for (size_t r = 0; r < result_indices.size(); ++r) {
                    const int64_t j   = result_indices[r];
                    double        phi = 0.0;
                    if (iObj.argAnal == AnalysisType::Voltage && pv != nullptr) {
                        phi = pv[j];
                    } else {
                        phi = (p1 ? p1[j] : 0.0) - (p2 ? p2[j] : 0.0);
                    }
                    trk.step(t[j], phi, iObj.argEventMetrics);
                }
                trk.finalize(iObj.argEventMetrics);
                outTrialEvents[b][k] = std::move(trk.events);
            }
        }
    } else {
        for (size_t k = 0; k < nPhase; ++k) {
            const auto& tr        = (*traceListPtr)[outPhaseTraces[k].relTraceIdx];
            int64_t     singleRow = -1;
            if (tr.variableIndex.has_value() && !tr.index1.has_value() && !tr.index2.has_value()) {
                singleRow = *tr.variableIndex;
            } else if (tr.index1.has_value() && !tr.index2.has_value() && !tr.variableIndex.has_value()) {
                singleRow = *tr.index1;
            } else if (!tr.index1.has_value() && tr.index2.has_value() && !tr.variableIndex.has_value()) {
                singleRow = *tr.index2;
            }
            if (sObj.eventsOnly_ && singleRow >= 0
                && singleRow < static_cast<int64_t>(sObj.useEventTracker_.size())
                && sObj.useEventTracker_[singleRow]) {
                outTrialEvents[0][k] = sObj.eventTrackers_[singleRow].events;
                continue;
            }
            const double* p1 = (tr.index1 && *tr.index1 >= 0 && *tr.index1 < static_cast<int64_t>(x.size()) && x[*tr.index1])
                                       ? x[*tr.index1]->data()
                                       : nullptr;
            const double* p2 = (tr.index2 && *tr.index2 >= 0 && *tr.index2 < static_cast<int64_t>(x.size()) && x[*tr.index2])
                                       ? x[*tr.index2]->data()
                                       : nullptr;
            const double* pv = (tr.variableIndex && *tr.variableIndex >= 0
                                && *tr.variableIndex < static_cast<int64_t>(x.size()) && x[*tr.variableIndex])
                                       ? x[*tr.variableIndex]->data()
                                       : nullptr;
            if (result_indices.empty()) { continue; }
            InLoopEventTracker trk;
            trk.traceId = static_cast<int32_t>(k);
            trk.sign    = 1;
            trk.init_metrics(iObj.argEventMetrics);
            for (size_t r = 0; r < result_indices.size(); ++r) {
                const int64_t j   = result_indices[r];
                double        phi = 0.0;
                if (iObj.argAnal == AnalysisType::Voltage && pv != nullptr) {
                    phi = pv[j];
                } else {
                    phi = (p1 ? p1[j] : 0.0) - (p2 ? p2[j] : 0.0);
                }
                trk.step(t[j], phi, iObj.argEventMetrics);
            }
            trk.finalize(iObj.argEventMetrics);
            outTrialEvents[0][k] = std::move(trk.events);
        }
    }
}
} // namespace

void Output::write_margins(const Input& iObj, Matrix& mObj, Simulation& sObj) {
    if (iObj.marginTargets.empty()) { return; }

    std::vector<PhaseTraceInfo>                     phaseTraces;
    std::vector<std::vector<std::vector<SfqEvent>>> trialEvents;
    extract_trial_phase_events(iObj, mObj, sObj, phaseTraces, trialEvents);
    if (phaseTraces.empty() || trialEvents.empty()) { return; }

    const auto& nomEvents      = trialEvents[0];
    int64_t     nomTotalPulses = 0;
    for (const auto& evs : nomEvents) { nomTotalPulses += static_cast<int64_t>(evs.size()); }

    struct MarginRow {
        size_t            targetIdx       = 0;
        std::string       name;
        std::string       category;
        double            nominalValue    = 1.0;
        double            minPassScale    = 1.0;
        double            maxPassScale    = 1.0;
        double            marginLowPct    = 0.0;
        double            marginHighPct   = 0.0;
        double            critMarginPct   = 0.0;
        double            centerOffsetPct = 0.0;
        bool              lowBounded      = false;
        bool              highBounded     = false;
        int64_t           passPoints      = 0;
        int64_t           totalPoints     = 0;
        double            maxPassSkewSec  = 0.0;
        EventSeqDiagnosis lowFailDiag;
        EventSeqDiagnosis highFailDiag;
        std::string       limitingSide;
        std::string       limitingModeStr;
    };

    std::vector<MarginRow> rows;
    rows.reserve(iObj.marginTargets.size());

    for (size_t tIdx = 0; tIdx < iObj.marginTargets.size(); ++tIdx) {
        const auto& mt = iObj.marginTargets[tIdx];
        const size_t P = mt.values.size();
        if (P == 0 || mt.trialIndices.size() != P) { continue; }

        std::vector<EventSeqDiagnosis> diags(P);
        for (size_t p = 0; p < P; ++p) {
            const int64_t b = mt.trialIndices[p];
            if (b == 0) {
                diags[p].passed      = true;
                diags[p].totalPulses = nomTotalPulses;
            } else if (b > 0 && static_cast<size_t>(b) < trialEvents.size()) {
                diags[p] = compare_event_sequences(
                        phaseTraces, nomEvents, trialEvents[b], iObj.argMarginTol, iObj.transSim.tstop());
            } else {
                diags[p].passed = false;
                diags[p].kind   = "SIM_ERROR";
            }
        }

        const size_t nomIdx = std::min(mt.nomPointIdx, P - 1);
        size_t       loIdx  = nomIdx;
        size_t       hiIdx  = nomIdx;
        while (loIdx > 0 && diags[loIdx - 1].passed) { --loIdx; }
        while (hiIdx + 1 < P && diags[hiIdx + 1].passed) { ++hiIdx; }

        MarginRow r;
        r.targetIdx       = tIdx;
        r.name            = mt.displayName;
        r.category        = mt.category;
        r.nominalValue    = mt.nominalValue;
        r.minPassScale    = mt.values[loIdx];
        r.maxPassScale    = mt.values[hiIdx];
        r.marginLowPct    = (1.0 - r.minPassScale) * 100.0;
        r.marginHighPct   = (r.maxPassScale - 1.0) * 100.0;
        r.critMarginPct   = std::min(r.marginLowPct, r.marginHighPct);
        r.centerOffsetPct = (0.5 * (r.minPassScale + r.maxPassScale) - 1.0) * 100.0;
        r.lowBounded      = (loIdx > 0);
        r.highBounded     = (hiIdx + 1 < P);
        r.passPoints      = static_cast<int64_t>(hiIdx - loIdx + 1);
        r.totalPoints     = static_cast<int64_t>(P);

        for (size_t p = loIdx; p <= hiIdx; ++p) {
            if (diags[p].maxTimingSkew > r.maxPassSkewSec) {
                r.maxPassSkewSec = diags[p].maxTimingSkew;
            }
        }
        if (r.lowBounded) { r.lowFailDiag = diags[loIdx - 1]; }
        if (r.highBounded) { r.highFailDiag = diags[hiIdx + 1]; }

        if (r.lowBounded && (!r.highBounded || r.marginLowPct <= r.marginHighPct)) {
            r.limitingSide    = "LOWER";
            r.limitingModeStr = "[-]: " + format_diagnosis_short(r.lowFailDiag);
        } else if (r.highBounded) {
            r.limitingSide    = "UPPER";
            r.limitingModeStr = "[+]: " + format_diagnosis_short(r.highFailDiag);
        } else {
            r.limitingSide    = "NONE";
            r.limitingModeStr = "Exceeds tested range [" + Misc::precise_to_string(mt.values.front()) + ", "
                                + Misc::precise_to_string(mt.values.back()) + "]";
        }

        rows.push_back(std::move(r));
    }

    std::stable_sort(rows.begin(), rows.end(), [](const MarginRow& a, const MarginRow& b) {
        const bool aBounded = (a.lowBounded || a.highBounded);
        const bool bBounded = (b.lowBounded || b.highBounded);
        if (aBounded != bBounded) { return aBounded > bBounded; }
        if (std::abs(a.critMarginPct - b.critMarginPct) > 1e-6) {
            return a.critMarginPct < b.critMarginPct;
        }
        return a.targetIdx < b.targetIdx;
    });

    if (!iObj.argMin && !rows.empty()) {
        std::cout << "\n========================================================================================================\n";
        std::cout << "JoSIM Critical Operating Margin Report (" << rows.size() << " target(s), "
                  << sObj.results.mcRuns << " batched trials, " << phaseTraces.size() << " JJs, "
                  << nomTotalPulses << " nominal SFQ events, tol = +/-" << std::fixed << std::setprecision(1)
                  << (iObj.argMarginTol * 1e12) << " ps)\n";
        std::cout << "========================================================================================================\n";
        std::cout << std::left << std::setw(5) << "Rank" << std::setw(18) << "Target" << std::setw(8) << "Cat"
                  << std::right << std::setw(10) << "Lower" << std::setw(10) << "Upper" << std::setw(17)
                  << "Window [Scale]" << std::setw(10) << "Center" << std::setw(10) << "Crit(%)"
                  << "  Limiting Failure Mode (Root Cause)\n";
        std::cout << "--------------------------------------------------------------------------------------------------------\n";

        for (size_t idx = 0; idx < rows.size(); ++idx) {
            const auto& r = rows[idx];
            std::ostringstream loStr, hiStr, winStr, ctrStr, critStr;
            loStr << (r.lowBounded ? "-" : "<-") << std::fixed << std::setprecision(1) << r.marginLowPct << "%";
            hiStr << (r.highBounded ? "+" : ">+") << std::fixed << std::setprecision(1) << r.marginHighPct << "%";
            winStr << "[" << std::fixed << std::setprecision(2) << r.minPassScale << ", " << r.maxPassScale << "]";
            ctrStr << std::showpos << std::fixed << std::setprecision(1) << r.centerOffsetPct << "%" << std::noshowpos;
            critStr << ((!r.lowBounded && !r.highBounded) ? ">" : "") << std::fixed << std::setprecision(1)
                    << r.critMarginPct << "%";

            std::cout << std::left << std::setw(5) << (idx + 1) << std::setw(18) << r.name << std::setw(8)
                      << r.category << std::right << std::setw(10) << loStr.str() << std::setw(10) << hiStr.str()
                      << std::setw(17) << winStr.str() << std::setw(10) << ctrStr.str() << std::setw(10)
                      << critStr.str() << "  " << r.limitingModeStr << "\n";
        }
        std::cout << "--------------------------------------------------------------------------------------------------------\n";

        const auto& top = rows.front();
        std::cout << "  Critical Bottleneck : " << top.name << " (" << top.category << ") -> " << std::fixed
                  << std::setprecision(1) << top.critMarginPct << "% on " << top.limitingSide << " boundary ("
                  << top.limitingModeStr << ")\n";
        if (top.lowBounded && top.highBounded && std::abs(top.centerOffsetPct) >= 1.5) {
            const double symPct   = 0.5 * (top.marginLowPct + top.marginHighPct);
            const double ctrScale = 0.5 * (top.minPassScale + top.maxPassScale);
            std::cout << "  Centering Advisory  : " << top.name << " is off-center by " << std::showpos << std::fixed
                      << std::setprecision(1) << top.centerOffsetPct << "%" << std::noshowpos << " (scale "
                      << std::setprecision(3) << ctrScale << "x";
            if (std::abs(top.nominalValue - 1.0) > 1e-12 && std::abs(top.nominalValue) > 1e-30) {
                std::cout << ", value " << std::scientific << std::setprecision(3) << (top.nominalValue * ctrScale);
            }
            std::cout << "); recentering improves critical margin from " << std::fixed << std::setprecision(1)
                      << top.critMarginPct << "% -> +/-" << symPct << "%.\n";
        }
        std::cout << "========================================================================================================\n\n";
    }

    if (iObj.marginFile.has_value() && !iObj.marginFile->empty()) {
        std::ofstream outfile(*iObj.marginFile);
        if (!outfile.is_open()) {
            Errors::output_errors(OutputErrors::CANNOT_OPEN_FILE, *iObj.marginFile);
            return;
        }
        outfile << "rank,target,category,nominal_value,min_pass_scale,max_pass_scale,margin_low_pct,margin_high_pct,"
                   "crit_margin_pct,center_offset_pct,low_bounded,high_bounded,limiting_side,low_fail_kind,"
                   "low_fail_trace,low_fail_pulse,low_fail_time_s,high_fail_kind,high_fail_trace,high_fail_pulse,"
                   "high_fail_time_s,max_pass_timing_skew_s\n";
        for (size_t idx = 0; idx < rows.size(); ++idx) {
            const auto& r = rows[idx];
            outfile << (idx + 1) << "," << r.name << "," << r.category << "," << std::scientific
                    << std::setprecision(6) << r.nominalValue << "," << std::fixed << std::setprecision(4)
                    << r.minPassScale << "," << r.maxPassScale << "," << std::setprecision(2) << r.marginLowPct << ","
                    << r.marginHighPct << "," << r.critMarginPct << "," << r.centerOffsetPct << ","
                    << (r.lowBounded ? 1 : 0) << "," << (r.highBounded ? 1 : 0) << "," << r.limitingSide << ","
                    << (r.lowBounded ? r.lowFailDiag.kind : "NONE") << ","
                    << (r.lowBounded ? r.lowFailDiag.traceLabel : "") << ","
                    << (r.lowBounded ? r.lowFailDiag.pulseIndex : 0) << "," << std::scientific << std::setprecision(6)
                    << (r.lowBounded ? r.lowFailDiag.failTime : 0.0) << ","
                    << (r.highBounded ? r.highFailDiag.kind : "NONE") << ","
                    << (r.highBounded ? r.highFailDiag.traceLabel : "") << ","
                    << (r.highBounded ? r.highFailDiag.pulseIndex : 0) << "," << std::scientific
                    << std::setprecision(6) << (r.highBounded ? r.highFailDiag.failTime : 0.0) << ","
                    << r.maxPassSkewSec << "\n";
        }
    }
}

void Output::verify_golden_events(const Input& iObj, Matrix& mObj, Simulation& sObj) {
    if (!iObj.verifyEventsFile.has_value() || iObj.verifyEventsFile->empty()) { return; }
    const std::string& filename = *iObj.verifyEventsFile;

    std::vector<PhaseTraceInfo>                     phaseTraces;
    std::vector<std::vector<std::vector<SfqEvent>>> trialEvents;
    extract_trial_phase_events(iObj, mObj, sObj, phaseTraces, trialEvents);
    if (phaseTraces.empty() || trialEvents.empty()) { return; }

    std::unordered_map<std::string, std::vector<SfqEvent>> goldenByTrace;
    std::string ext = std::filesystem::path(filename).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".npy") {
        std::ifstream infile(filename, std::ios::in | std::ios::binary);
        if (!infile.is_open()) {
            Errors::output_errors(OutputErrors::CANNOT_OPEN_FILE, filename);
            return;
        }
        char magic[6] = {0};
        infile.read(magic, 6);
        uint8_t major = 0, minor = 0;
        infile.read(reinterpret_cast<char*>(&major), 1);
        infile.read(reinterpret_cast<char*>(&minor), 1);
        uint32_t headerLen = 0;
        if (major <= 1) {
            uint16_t h16 = 0;
            infile.read(reinterpret_cast<char*>(&h16), 2);
            headerLen = h16;
        } else {
            infile.read(reinterpret_cast<char*>(&headerLen), 4);
        }
        std::string hdr(headerLen, '\0');
        infile.read(hdr.data(), headerLen);

        size_t strLen = 1;
        auto   posS   = hdr.find("|S");
        if (posS != std::string::npos) {
            strLen = static_cast<size_t>(std::strtoull(hdr.c_str() + posS + 2, nullptr, 10));
        }
        size_t nRows = 0;
        auto   posSh = hdr.find("'shape': (");
        if (posSh != std::string::npos) {
            nRows = static_cast<size_t>(std::strtoull(hdr.c_str() + posSh + 10, nullptr, 10));
        }
        size_t extraCols = 0;
        for (const char* colName : {"'t_peak'", "'v_peak'", "'fwhm'", "'dt_pi'", "'area'"}) {
            if (hdr.find(colName) != std::string::npos) { ++extraCols; }
        }
        const size_t      recSize = 4 + strLen + 8 + 4 + 4 + 8 + 8 * extraCols;
        std::vector<char> rec(recSize, 0);
        for (size_t r = 0; r < nRows; ++r) {
            if (!infile.read(rec.data(), static_cast<std::streamsize>(recSize))) { break; }
            std::string rawName(rec.data() + 4, strLen);
            const auto  nulPos = rawName.find('\0');
            if (nulPos != std::string::npos) { rawName.resize(nulPos); }
            std::string normName = normalize_event_trace_label(rawName);
            const auto  mcPos    = normName.rfind("_MC0");
            if (mcPos != std::string::npos && mcPos + 4 == normName.size()) { normName.resize(mcPos); }

            SfqEvent ev;
            std::memcpy(&ev.traceId, rec.data(), 4);
            std::memcpy(&ev.time, rec.data() + 4 + strLen, 8);
            std::memcpy(&ev.polarity, rec.data() + 4 + strLen + 8, 4);
            std::memcpy(&ev.winding, rec.data() + 4 + strLen + 12, 4);
            std::memcpy(&ev.phase, rec.data() + 4 + strLen + 16, 8);
            goldenByTrace[normName].push_back(ev);
        }
    } else {
        std::ifstream infile(filename);
        if (!infile.is_open()) {
            Errors::output_errors(OutputErrors::CANNOT_OPEN_FILE, filename);
            return;
        }
        std::string line;
        while (std::getline(infile, line)) {
            if (line.empty()) { continue; }
            std::vector<std::string> cols;
            std::stringstream        ss(line);
            std::string              cell;
            while (std::getline(ss, cell, ',')) { cols.push_back(cell); }
            if (cols.size() < 4) { continue; }
            if (cols[0].find("trace") != std::string::npos || cols[0].find("TRACE") != std::string::npos
                || cols[0].front() == '#') {
                continue;
            }
            std::string normName = normalize_event_trace_label(cols[1]);
            const auto  mcPos    = normName.rfind("_MC0");
            if (mcPos != std::string::npos && mcPos + 4 == normName.size()) { normName.resize(mcPos); }

            SfqEvent ev;
            try {
                ev.traceId  = static_cast<int32_t>(std::stol(cols[0]));
                ev.time     = std::stod(cols[2]);
                ev.polarity = static_cast<int32_t>(std::stol(cols[3]));
                if (cols.size() >= 5) { ev.winding = static_cast<int32_t>(std::stol(cols[4])); }
                if (cols.size() >= 6) { ev.phase = std::stod(cols[5]); }
            } catch (...) { continue; }
            goldenByTrace[normName].push_back(ev);
        }
    }

    for (auto& kv : goldenByTrace) {
        std::stable_sort(kv.second.begin(), kv.second.end(), [](const SfqEvent& a, const SfqEvent& b) {
            return a.time < b.time;
        });
    }

    std::vector<std::vector<SfqEvent>> refEvents(phaseTraces.size());
    for (size_t k = 0; k < phaseTraces.size(); ++k) {
        const std::string normLbl = normalize_event_trace_label(phaseTraces[k].label);
        auto              it      = goldenByTrace.find(normLbl);
        if (it == goldenByTrace.end() && normLbl.rfind("P(", 0) == 0 && normLbl.back() == ')') {
            it = goldenByTrace.find(normLbl.substr(2, normLbl.size() - 3));
        }
        if (it != goldenByTrace.end()) {
            refEvents[k] = it->second;
        }
    }

    const EventSeqDiagnosis diag = compare_event_sequences(
            phaseTraces, refEvents, trialEvents[0], iObj.argMarginTol, iObj.transSim.tstop());
    if (diag.passed) {
        if (!iObj.argMin) {
            std::cout << "JoSIM Event Verification: PASS (" << diag.totalPulses << " events across "
                      << phaseTraces.size() << " traces matched '" << filename << "' within +/-" << std::fixed
                      << std::setprecision(1) << (iObj.argMarginTol * 1e12) << " ps, max skew = " << std::setprecision(2)
                      << (diag.maxTimingSkew * 1e12) << " ps)\n";
        }
    } else {
        std::cerr << "JoSIM Event Verification: FAIL against '" << filename << "' (tol = +/-" << std::fixed
                  << std::setprecision(1) << (iObj.argMarginTol * 1e12) << " ps)\n"
                  << "  Root-Cause Failure: " << format_diagnosis_short(diag) << "\n";
        std::exit(1);
    }
}

void Output::verify_assertions(const Input& iObj, Matrix& mObj, Simulation& sObj) {
    if (iObj.assertions.empty()) { return; }

    std::vector<PhaseTraceInfo>                     phaseTraces;
    std::vector<std::vector<std::vector<SfqEvent>>> trialEvents;
    extract_trial_phase_events(iObj, mObj, sObj, phaseTraces, trialEvents);
    if (trialEvents.empty()) { return; }

    auto assert_glob_match = [](const std::string& text, const std::string& pat) -> bool {
        size_t ti = 0, pi = 0, starIdx = std::string::npos, matchIdx = 0;
        while (ti < text.size()) {
            if (pi < pat.size() && (pat[pi] == '?' || pat[pi] == text[ti])) {
                ++ti;
                ++pi;
            } else if (pi < pat.size() && pat[pi] == '*') {
                starIdx  = pi++;
                matchIdx = ti;
            } else if (starIdx != std::string::npos) {
                pi = starIdx + 1;
                ti = ++matchIdx;
            } else {
                return false;
            }
        }
        while (pi < pat.size() && pat[pi] == '*') { ++pi; }
        return pi == pat.size();
    };

    auto find_trace_indices = [&](std::string rawName) -> std::vector<int64_t> {
        std::vector<int64_t> out;
        std::string          norm = normalize_event_trace_label(rawName);
        std::replace(norm.begin(), norm.end(), '.', '|');
        std::string inner = norm;
        if (inner.size() > 3 && (inner.rfind("P(", 0) == 0 || inner.rfind("V(", 0) == 0) && inner.back() == ')') {
            inner = inner.substr(2, inner.size() - 3);
        }
        const std::string withP   = "P(" + inner + ")";
        const bool        hasGlob = (inner.find('*') != std::string::npos || inner.find('?') != std::string::npos);
        for (size_t k = 0; k < phaseTraces.size(); ++k) {
            std::string ptNorm = normalize_event_trace_label(phaseTraces[k].label);
            std::replace(ptNorm.begin(), ptNorm.end(), '.', '|');
            std::string ptInner = ptNorm;
            if (ptInner.size() > 3 && ptInner.rfind("P(", 0) == 0 && ptInner.back() == ')') {
                ptInner = ptInner.substr(2, ptInner.size() - 3);
            }
            if (!hasGlob) {
                if (ptNorm == withP || ptNorm == inner) {
                    out.push_back(static_cast<int64_t>(k));
                    break;
                }
            } else {
                if (assert_glob_match(ptInner, inner) || assert_glob_match(ptNorm, withP)) {
                    out.push_back(static_cast<int64_t>(k));
                }
            }
        }
        return out;
    };

    auto find_trace_idx = [&](std::string rawName) -> int64_t {
        const auto v = find_trace_indices(rawName);
        return v.empty() ? -1 : v.front();
    };

    const size_t             nTrialsToCheck = (iObj.argMargins || iObj.argSens) ? 1 : trialEvents.size();
    std::vector<std::string> failures;
    size_t                   totalPassed = 0;
    size_t                   totalChecks = 0;

    for (size_t b = 0; b < nTrialsToCheck; ++b) {
        const auto& curTrialEvs = trialEvents[b];
        for (size_t aIdx = 0; aIdx < iObj.assertions.size(); ++aIdx) {
            const auto&                as       = iObj.assertions[aIdx];
            const std::vector<int64_t> tIndices = find_trace_indices(as.trace);
            if (tIndices.empty()) {
                ++totalChecks;
                std::ostringstream oss;
                oss << "Assertion #" << (aIdx + 1) << " [" << as.trace << "]: trace not found in phase traces";
                failures.push_back(oss.str());
                continue;
            }
            for (const int64_t tIdx : tIndices) {
                ++totalChecks;
                const auto&       evs      = curTrialEvs[static_cast<size_t>(tIdx)];
                const std::string traceLbl = phaseTraces[static_cast<size_t>(tIdx)].label;

            if (as.kind == AssertKind::BITSTREAM) {
                if (as.per <= 0.0 || as.bits.empty()) {
                    std::ostringstream oss;
                    oss << "Assertion #" << (aIdx + 1) << " [" << traceLbl
                        << "]: invalid BITSTREAM parameters (PER=" << as.per << ", BITS='" << as.bits << "')";
                    failures.push_back(oss.str());
                    continue;
                }
                const double      win     = (as.window > 0.0) ? as.window : as.per;
                const double      tol     = (as.tol > 0.0) ? as.tol : 0.0;
                const size_t      nBits   = as.bits.size();
                std::vector<bool> claimed(evs.size(), false);
                std::string       actualBits(nBits, '0');

                // Pass 1: claim events that fall strictly inside nominal bit window [t0, t0 + win)
                std::vector<std::vector<size_t>> winEvs(nBits);
                for (size_t k = 0; k < nBits; ++k) {
                    const double t0 = as.td + static_cast<double>(k) * as.per;
                    const double t1 = t0 + win;
                    for (size_t e = 0; e < evs.size(); ++e) {
                        if (!claimed[e] && evs[e].time >= t0 && evs[e].time < t1) {
                            claimed[e] = true;
                            winEvs[k].push_back(e);
                        }
                    }
                }
                // Pass 2: if tol > 0, allow windows expecting a pulse ('1' or '-') with 0 claimed events
                // to claim an unclaimed event within [t0 - tol, t0 + win + tol)
                if (tol > 0.0) {
                    for (size_t k = 0; k < nBits; ++k) {
                        const char expC = as.bits[k];
                        if ((expC == '1' || expC == '-') && winEvs[k].empty()) {
                            const double t0 = as.td + static_cast<double>(k) * as.per - tol;
                            const double t1 = as.td + static_cast<double>(k) * as.per + win + tol;
                            for (size_t e = 0; e < evs.size(); ++e) {
                                if (!claimed[e] && evs[e].time >= t0 && evs[e].time < t1) {
                                    claimed[e] = true;
                                    winEvs[k].push_back(e);
                                    break;
                                }
                            }
                        }
                    }
                }

                int64_t firstFailBit = -1;
                for (size_t k = 0; k < nBits; ++k) {
                    const auto& w = winEvs[k];
                    if (w.empty()) {
                        actualBits[k] = '0';
                    } else if (w.size() == 1) {
                        actualBits[k] = (evs[w[0]].polarity >= 0) ? '1' : '-';
                    } else {
                        actualBits[k] = 'M';
                    }
                    const char expC = as.bits[k];
                    if (expC == 'X' || expC == '?') { continue; }
                    if (actualBits[k] != expC && firstFailBit < 0) { firstFailBit = static_cast<int64_t>(k); }
                }

                if (firstFailBit < 0) {
                    ++totalPassed;
                } else {
                    const double       fT0 = (as.td + static_cast<double>(firstFailBit) * as.per) * 1e12;
                    const double       fT1 = (as.td + static_cast<double>(firstFailBit) * as.per + win) * 1e12;
                    std::ostringstream oss;
                    oss << "Assertion #" << (aIdx + 1) << " [" << traceLbl << " BITSTREAM";
                    if (nTrialsToCheck > 1) { oss << " trial #" << b; }
                    oss << "] FAILED at bit " << firstFailBit << " [" << std::fixed << std::setprecision(1) << fT0
                        << ".." << fT1 << " ps]: expected '" << as.bits[static_cast<size_t>(firstFailBit)]
                        << "', got '" << actualBits[static_cast<size_t>(firstFailBit)] << "' (expected=" << as.bits
                        << ", actual=" << actualBits << ")";
                    failures.push_back(oss.str());
                }
            } else if (as.kind == AssertKind::COUNT) {
                int64_t cnt = 0;
                for (const auto& ev : evs) {
                    if (as.tFrom >= 0.0 && ev.time < as.tFrom) { continue; }
                    if (as.tTo >= 0.0 && ev.time > as.tTo) { continue; }
                    ++cnt;
                }
                if (cnt == as.expCount) {
                    ++totalPassed;
                } else {
                    std::ostringstream oss;
                    oss << "Assertion #" << (aIdx + 1) << " [" << traceLbl << " COUNT";
                    if (nTrialsToCheck > 1) { oss << " trial #" << b; }
                    oss << "] FAILED: expected " << as.expCount << " pulses, got " << cnt;
                    failures.push_back(oss.str());
                }
            } else if (as.kind == AssertKind::TIMES) {
                const double tol  = (as.tol >= 0.0) ? as.tol : iObj.argMarginTol;
                bool         ok   = (evs.size() == as.expTimes.size());
                size_t       badP = 0;
                for (size_t p = 0; p < std::min(evs.size(), as.expTimes.size()); ++p) {
                    if (std::abs(evs[p].time - as.expTimes[p]) > tol) {
                        ok   = false;
                        badP = p;
                        break;
                    }
                }
                if (ok) {
                    ++totalPassed;
                } else {
                    std::ostringstream oss;
                    oss << "Assertion #" << (aIdx + 1) << " [" << traceLbl << " TIMES";
                    if (nTrialsToCheck > 1) { oss << " trial #" << b; }
                    oss << "] FAILED: ";
                    if (evs.size() != as.expTimes.size()) {
                        oss << "expected " << as.expTimes.size() << " pulses, got " << evs.size();
                    } else {
                        const double dtPs = (evs[badP].time - as.expTimes[badP]) * 1e12;
                        oss << "pulse #" << (badP + 1) << " expected @ " << std::fixed << std::setprecision(2)
                            << (as.expTimes[badP] * 1e12) << " ps, got " << (evs[badP].time * 1e12) << " ps (delta="
                            << std::showpos << dtPs << std::noshowpos << " ps, tol=+/-" << (tol * 1e12) << " ps)";
                    }
                    failures.push_back(oss.str());
                }
            } else if (as.kind == AssertKind::AFTER) {
                const int64_t rIdx = find_trace_idx(as.refTrace);
                if (rIdx < 0) {
                    std::ostringstream oss;
                    oss << "Assertion #" << (aIdx + 1) << " [" << traceLbl << " AFTER " << as.refTrace
                        << "]: reference trace not found in phase traces";
                    failures.push_back(oss.str());
                    continue;
                }
                const auto&        refEvs = curTrialEvs[static_cast<size_t>(rIdx)];
                const std::string& refLbl = phaseTraces[static_cast<size_t>(rIdx)].label;
                const double       tol    = (as.tol >= 0.0) ? as.tol : iObj.argMarginTol;
                bool               ok     = (evs.size() == refEvs.size());
                size_t             badP   = 0;
                for (size_t p = 0; p < std::min(evs.size(), refEvs.size()); ++p) {
                    const double actualDelay = evs[p].time - refEvs[p].time;
                    if (evs[p].polarity != refEvs[p].polarity || std::abs(actualDelay - as.delay) > tol) {
                        ok   = false;
                        badP = p;
                        break;
                    }
                }
                if (ok) {
                    ++totalPassed;
                } else {
                    std::ostringstream oss;
                    oss << "Assertion #" << (aIdx + 1) << " [" << traceLbl << " AFTER " << refLbl;
                    if (nTrialsToCheck > 1) { oss << " trial #" << b; }
                    oss << "] FAILED: ";
                    if (evs.size() != refEvs.size()) {
                        oss << "pulse count mismatch (" << traceLbl << "=" << evs.size() << ", " << refLbl << "="
                            << refEvs.size() << ")";
                    } else {
                        const double actualDelayPs = (evs[badP].time - refEvs[badP].time) * 1e12;
                        const double expDelayPs    = as.delay * 1e12;
                        oss << "pulse #" << (badP + 1) << " delay=" << std::fixed << std::setprecision(2)
                            << actualDelayPs << " ps, expected " << expDelayPs << " +/- " << (tol * 1e12) << " ps";
                    }
                    failures.push_back(oss.str());
                }
            }
            }
        }
    }

    if (failures.empty()) {
        if (!iObj.argMin) {
            std::cout << "JoSIM Assertions: PASS (" << totalPassed << "/" << totalChecks << " assertions passed";
            if (nTrialsToCheck > 1) { std::cout << " across " << nTrialsToCheck << " trials"; }
            std::cout << ")\n";
        }
    } else {
        std::cerr << "JoSIM Assertions: FAIL (" << failures.size() << " of " << totalChecks << " assertions failed)\n";
        for (const auto& f : failures) { std::cerr << "  " << f << "\n"; }
        std::exit(1);
    }
}

void Output::evaluate_measurements(const Input& iObj, Matrix& mObj, Simulation& sObj) {
    if (iObj.measurements.empty()) { return; }

    std::vector<PhaseTraceInfo>                     phaseTraces;
    std::vector<std::vector<std::vector<SfqEvent>>> trialEvents;
    extract_trial_phase_events(iObj, mObj, sObj, phaseTraces, trialEvents);

    const auto& x    = sObj.results.xVector;
    const auto& t    = sObj.results.timeAxis;
    const auto& tran = iObj.transSim;

    std::vector<int64_t> result_indices;
    if (!sObj.results.printIndices.empty()) {
        result_indices = sObj.results.printIndices;
    } else if (!t.empty()) {
        for (size_t i = 0; i < t.size(); ++i) {
            if (t[i] >= tran.prstart()) {
                result_indices.emplace_back(static_cast<int64_t>(i));
                break;
            }
        }
        double next_print_point = tran.prstart() + tran.prstep();
        if (!result_indices.empty()) {
            for (size_t i = static_cast<size_t>(result_indices.back()); i < t.size(); ++i) {
                if (t[i] >= next_print_point) {
                    result_indices.emplace_back(static_cast<int64_t>(i));
                    next_print_point += tran.prstep();
                }
            }
        }
    }

    std::vector<double> tGrid;
    tGrid.reserve(result_indices.size());
    for (int64_t j : result_indices) {
        if (j >= 0 && static_cast<size_t>(j) < t.size()) { tGrid.push_back(t[static_cast<size_t>(j)]); }
    }

    const int64_t halffirsize = static_cast<int64_t>(tran.prfirwindow() <= 0. ? 0 : .5 * tran.prfirwindow() / tran.tstep());
    const int64_t firsize     = halffirsize * 2 + 1;
    std::vector<double> firwindow(static_cast<size_t>(firsize), 1.0);
    if (firsize > 1) {
        double firsum = 0.0;
        for (int64_t k = 0; k < firsize; ++k) {
            const double hanning        = 1.0 - std::cos((static_cast<double>(k) + 0.5) * (2.0 * Constants::PI / static_cast<double>(firsize)));
            firsum                     += hanning;
            firwindow[static_cast<size_t>(k)] = hanning;
        }
        const double firscale = 1.0 / firsum;
        for (double& w : firwindow) { w *= firscale; }
    }

    auto find_phase_event_idx = [&](std::string rawName) -> int64_t {
        std::string norm = normalize_event_trace_label(rawName);
        std::replace(norm.begin(), norm.end(), '.', '|');
        std::string inner = norm;
        if (inner.size() > 3 && (inner.rfind("P(", 0) == 0 || inner.rfind("V(", 0) == 0 || inner.rfind("I(", 0) == 0)
            && inner.back() == ')') {
            inner = inner.substr(2, inner.size() - 3);
        }
        const std::string withP = "P(" + inner + ")";
        for (size_t k = 0; k < phaseTraces.size(); ++k) {
            std::string ptNorm = normalize_event_trace_label(phaseTraces[k].label);
            std::replace(ptNorm.begin(), ptNorm.end(), '.', '|');
            if (ptNorm == withP || ptNorm == inner) { return static_cast<int64_t>(k); }
        }
        return -1;
    };

    auto find_rel_trace_idx = [&](std::string rawName, bool forcePhase) -> int64_t {
        std::string norm = normalize_event_trace_label(rawName);
        std::replace(norm.begin(), norm.end(), '.', '|');
        std::string inner  = norm;
        char        prefix = '\0';
        if (inner.size() > 3 && (inner.rfind("P(", 0) == 0 || inner.rfind("V(", 0) == 0 || inner.rfind("I(", 0) == 0)
            && inner.back() == ')') {
            prefix = inner[0];
            inner  = inner.substr(2, inner.size() - 3);
        }
        StorageType wantSt = (iObj.argAnal == AnalysisType::Voltage) ? StorageType::Voltage : StorageType::Phase;
        if (forcePhase || prefix == 'P') {
            wantSt = StorageType::Phase;
        } else if (prefix == 'V') {
            wantSt = StorageType::Voltage;
        } else if (prefix == 'I') {
            wantSt = StorageType::Current;
        }
        for (size_t k = 0; k < mObj.relevantTraces.size(); ++k) {
            const auto& rt = mObj.relevantTraces[k];
            if (rt.storageType != wantSt || !rt.deviceLabel.has_value()) { continue; }
            std::string rtNorm = normalize_event_trace_label(*rt.deviceLabel);
            std::replace(rtNorm.begin(), rtNorm.end(), '.', '|');
            std::string rtInner = rtNorm;
            if (rtInner.size() > 3
                && (rtInner.rfind("P(", 0) == 0 || rtInner.rfind("V(", 0) == 0 || rtInner.rfind("I(", 0) == 0)
                && rtInner.back() == ')') {
                rtInner = rtInner.substr(2, rtInner.size() - 3);
            }
            if (rtInner == inner || rtNorm == norm) { return static_cast<int64_t>(k); }
        }
        return -1;
    };

    auto eval_rel_trace_wave = [&](int64_t relIdx, int64_t b) -> std::vector<double> {
        std::vector<double> wave;
        if (relIdx < 0 || static_cast<size_t>(relIdx) >= mObj.relevantTraces.size() || tGrid.empty()) { return wave; }
        const auto&  rt     = mObj.relevantTraces[static_cast<size_t>(relIdx)];
        const auto   st     = rt.storageType;
        const size_t nPrint = tGrid.size();
        wave.resize(nPrint, 0.0);

        const int64_t i1 = rt.index1.has_value() ? *rt.index1 : -1;
        const int64_t i2 = rt.index2.has_value() ? *rt.index2 : -1;
        const int64_t vi = rt.variableIndex.has_value() ? *rt.variableIndex : -1;

        const bool    isISrc = (st == StorageType::Current && rt.deviceLabel.has_value() && rt.deviceLabel->size() > 3
                             && rt.deviceLabel->at(3) == 'I' && rt.sourceIndex.has_value());
        Function      localSrc;
        if (isISrc) { localSrc = mObj.sourcegen.at(*rt.sourceIndex); }

        if (sObj.results.mcRuns > 1 && !sObj.results.mcTraceOut.empty()) {
            const int64_t nRows = sObj.results.mcNumRows;
            const int64_t nKept = sObj.results.mcNumKeptSteps;
            const int32_t s1    = (i1 >= 0 && i1 < static_cast<int64_t>(sObj.results.mcRowToSlot.size()))
                                          ? sObj.results.mcRowToSlot[static_cast<size_t>(i1)]
                                          : -1;
            const int32_t s2    = (i2 >= 0 && i2 < static_cast<int64_t>(sObj.results.mcRowToSlot.size()))
                                          ? sObj.results.mcRowToSlot[static_cast<size_t>(i2)]
                                          : -1;
            const int32_t sv    = (vi >= 0 && vi < static_cast<int64_t>(sObj.results.mcRowToSlot.size()))
                                          ? sObj.results.mcRowToSlot[static_cast<size_t>(vi)]
                                          : -1;
            const double* p1    = (s1 >= 0) ? (sObj.results.mcTraceOut.data() + (b * nRows + s1) * nKept) : nullptr;
            const double* p2    = (s2 >= 0) ? (sObj.results.mcTraceOut.data() + (b * nRows + s2) * nKept) : nullptr;
            const double* pv    = (sv >= 0) ? (sObj.results.mcTraceOut.data() + (b * nRows + sv) * nKept) : nullptr;
            const size_t  xsize = std::min<size_t>(t.size(), static_cast<size_t>(nKept));
            for (size_t pIdx = 0; pIdx < nPrint; ++pIdx) {
                const int64_t j   = result_indices[pIdx];
                double        acc = 0.0;
                for (int64_t k = 0; k < firsize; ++k) {
                    int64_t jk = j + halffirsize - k;
                    if (jk < 0) {
                        jk = 0;
                    } else if (static_cast<size_t>(jk) >= xsize) {
                        jk = static_cast<int64_t>(xsize) - 1;
                    }
                    double value = 0.0;
                    if (st == StorageType::Voltage) {
                        const double valin1 = p1 ? p1[jk] : 0.0;
                        const double valin2 = p2 ? p2[jk] : 0.0;
                        if (iObj.argAnal == AnalysisType::Voltage) {
                            value = valin1 - valin2;
                        } else if (rt.deviceLabel.has_value() && rt.deviceLabel->size() > 3
                                   && rt.deviceLabel->at(3) == 'B' && pv) {
                            value = pv[jk];
                        } else {
                            double valin1n1, valin1n2, valin2n1, valin2n2;
                            if (jk >= 2) {
                                valin1n2 = p1 ? p1[jk - 2] : 0.0;
                                valin1n1 = p1 ? p1[jk - 1] : 0.0;
                                valin2n2 = p2 ? p2[jk - 2] : 0.0;
                                valin2n1 = p2 ? p2[jk - 1] : 0.0;
                            } else if (jk == 1) {
                                valin1n2 = valin1n1 = p1 ? p1[jk - 1] : 0.0;
                                valin2n2 = valin2n1 = p2 ? p2[jk - 1] : 0.0;
                            } else {
                                valin1n2 = valin1n1 = valin1;
                                valin2n2 = valin2n1 = valin2;
                            }
                            value = ((3.0 * Constants::SIGMA) / (2.0 * tran.tstep()))
                                    * ((valin1 - valin2) - (4.0 / 3.0) * (valin1n1 - valin2n1)
                                       + (1.0 / 3.0) * (valin1n2 - valin2n2));
                        }
                    } else if (st == StorageType::Phase) {
                        if (iObj.argAnal == AnalysisType::Phase) {
                            value = (p1 ? p1[jk] : 0.0) - (p2 ? p2[jk] : 0.0);
                        } else if (pv) {
                            value = pv[jk];
                        }
                    } else {
                        value = (!isISrc) ? (p1 ? p1[jk] : 0.0) : localSrc.value(t[static_cast<size_t>(jk)]);
                    }
                    acc += firwindow[static_cast<size_t>(k)] * value;
                }
                wave[pIdx] = acc;
            }
            return wave;
        }

        const std::vector<double>* v1 = (i1 >= 0 && static_cast<size_t>(i1) < x.size() && x[static_cast<size_t>(i1)])
                                                ? &(*x[static_cast<size_t>(i1)])
                                                : nullptr;
        const std::vector<double>* v2 = (i2 >= 0 && static_cast<size_t>(i2) < x.size() && x[static_cast<size_t>(i2)])
                                                ? &(*x[static_cast<size_t>(i2)])
                                                : nullptr;
        const std::vector<double>* vv = (vi >= 0 && static_cast<size_t>(vi) < x.size() && x[static_cast<size_t>(vi)])
                                                ? &(*x[static_cast<size_t>(vi)])
                                                : nullptr;
        size_t xsize = t.size();
        if (v1 && v1->size() < xsize) { xsize = v1->size(); }
        if (v2 && v2->size() < xsize) { xsize = v2->size(); }
        if (vv && vv->size() < xsize) { xsize = vv->size(); }
        if (xsize == 0) { return {}; }

        for (size_t pIdx = 0; pIdx < nPrint; ++pIdx) {
            const int64_t j   = result_indices[pIdx];
            double        acc = 0.0;
            for (int64_t k = 0; k < firsize; ++k) {
                int64_t jk = j + halffirsize - k;
                if (jk < 0) {
                    jk = 0;
                } else if (static_cast<size_t>(jk) >= xsize) {
                    jk = static_cast<int64_t>(xsize) - 1;
                }
                const size_t ujk   = static_cast<size_t>(jk);
                double       value = 0.0;
                if (st == StorageType::Voltage) {
                    const double valin1 = v1 ? (*v1)[ujk] : 0.0;
                    const double valin2 = v2 ? (*v2)[ujk] : 0.0;
                    if (iObj.argAnal == AnalysisType::Voltage) {
                        value = valin1 - valin2;
                    } else if (rt.deviceLabel.has_value() && rt.deviceLabel->size() > 3
                               && rt.deviceLabel->at(3) == 'B' && vv) {
                        value = (*vv)[ujk];
                    } else {
                        double valin1n1, valin1n2, valin2n1, valin2n2;
                        if (jk >= 2) {
                            valin1n2 = v1 ? (*v1)[ujk - 2] : 0.0;
                            valin1n1 = v1 ? (*v1)[ujk - 1] : 0.0;
                            valin2n2 = v2 ? (*v2)[ujk - 2] : 0.0;
                            valin2n1 = v2 ? (*v2)[ujk - 1] : 0.0;
                        } else if (jk == 1) {
                            valin1n2 = valin1n1 = v1 ? (*v1)[ujk - 1] : 0.0;
                            valin2n2 = valin2n1 = v2 ? (*v2)[ujk - 1] : 0.0;
                        } else {
                            valin1n2 = valin1n1 = valin1;
                            valin2n2 = valin2n1 = valin2;
                        }
                        value = ((3.0 * Constants::SIGMA) / (2.0 * tran.tstep()))
                                * ((valin1 - valin2) - (4.0 / 3.0) * (valin1n1 - valin2n1)
                                   + (1.0 / 3.0) * (valin1n2 - valin2n2));
                    }
                } else if (st == StorageType::Phase) {
                    if (iObj.argAnal == AnalysisType::Phase) {
                        value = (v1 ? (*v1)[ujk] : 0.0) - (v2 ? (*v2)[ujk] : 0.0);
                    } else if (vv) {
                        value = (*vv)[ujk];
                    }
                } else {
                    value = (!isISrc) ? (v1 ? (*v1)[ujk] : 0.0) : localSrc.value(t[ujk]);
                }
                acc += firwindow[static_cast<size_t>(k)] * value;
            }
            wave[pIdx] = acc;
        }
        return wave;
    };

    auto interp_wave_at = [&](const std::vector<double>& y, double tQuery) -> double {
        if (y.empty() || tGrid.empty() || y.size() != tGrid.size()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (tQuery <= tGrid.front()) { return y.front(); }
        if (tQuery >= tGrid.back()) { return y.back(); }
        auto it = std::lower_bound(tGrid.begin(), tGrid.end(), tQuery);
        if (it == tGrid.begin()) { return y.front(); }
        if (it == tGrid.end()) { return y.back(); }
        const size_t i1 = static_cast<size_t>(std::distance(tGrid.begin(), it));
        const size_t i0 = i1 - 1;
        const double dt = tGrid[i1] - tGrid[i0];
        if (dt <= 0.0) { return y[i0]; }
        const double alpha = (tQuery - tGrid[i0]) / dt;
        return y[i0] + alpha * (y[i1] - y[i0]);
    };

    auto deriv_wave_at = [&](const std::vector<double>& y, double tQuery) -> double {
        if (y.size() < 2 || tGrid.size() < 2 || y.size() != tGrid.size()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (tQuery <= tGrid.front()) {
            const double dt = tGrid[1] - tGrid[0];
            return (dt > 0.0) ? ((y[1] - y[0]) / dt) : 0.0;
        }
        if (tQuery >= tGrid.back()) {
            const size_t n  = tGrid.size();
            const double dt = tGrid[n - 1] - tGrid[n - 2];
            return (dt > 0.0) ? ((y[n - 1] - y[n - 2]) / dt) : 0.0;
        }
        auto         it = std::lower_bound(tGrid.begin(), tGrid.end(), tQuery);
        const size_t i1 = std::clamp<size_t>(static_cast<size_t>(std::distance(tGrid.begin(), it)), 1, tGrid.size() - 1);
        const size_t i0 = i1 - 1;
        const double dt = tGrid[i1] - tGrid[i0];
        return (dt > 0.0) ? ((y[i1] - y[i0]) / dt) : 0.0;
    };

    const int64_t B     = std::max<int64_t>(1, sObj.results.mcRuns);
    const size_t  nMeas = iObj.measurements.size();
    std::vector<std::vector<double>> measVals(
            nMeas, std::vector<double>(static_cast<size_t>(B), std::numeric_limits<double>::quiet_NaN()));

    for (int64_t b = 0; b < B; ++b) {
        std::unordered_map<int64_t, std::vector<double>> waveCache;
        auto get_wave = [&](const std::string& rawName, bool forcePhase) -> const std::vector<double>& {
            static const std::vector<double> emptyVec;
            const int64_t relIdx = find_rel_trace_idx(rawName, forcePhase);
            if (relIdx < 0) { return emptyVec; }
            auto it = waveCache.find(relIdx);
            if (it == waveCache.end()) {
                it = waveCache.emplace(relIdx, eval_rel_trace_wave(relIdx, b)).first;
            }
            return it->second;
        };

        auto eval_trigger = [&](const MeasTrigger& tr) -> double {
            if (!tr.active) { return std::numeric_limits<double>::quiet_NaN(); }
            if (tr.isFixedTime) { return tr.fixedTime; }
            if (tr.isSfqEvent) {
                const int64_t pIdx = find_phase_event_idx(tr.trace);
                if (pIdx < 0 || static_cast<size_t>(b) >= trialEvents.size()
                    || static_cast<size_t>(pIdx) >= trialEvents[static_cast<size_t>(b)].size()) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                const auto&           evs = trialEvents[static_cast<size_t>(b)][static_cast<size_t>(pIdx)];
                std::vector<SfqEvent> matched;
                for (const auto& ev : evs) {
                    if (ev.time < tr.td - 1e-18) { continue; }
                    if (tr.dir == MeasCrossDir::RISE && ev.polarity <= 0) { continue; }
                    if (tr.dir == MeasCrossDir::FALL && ev.polarity >= 0) { continue; }
                    matched.push_back(ev);
                }
                if (matched.empty()) { return std::numeric_limits<double>::quiet_NaN(); }
                if (tr.sfqIndex < 0) { return matched.back().time; }
                if (static_cast<size_t>(tr.sfqIndex) <= matched.size()) {
                    return matched[static_cast<size_t>(tr.sfqIndex - 1)].time;
                }
                return std::numeric_limits<double>::quiet_NaN();
            }

            const auto& y1 = get_wave(tr.trace, false);
            if (y1.size() < 2 || y1.size() != tGrid.size()) { return std::numeric_limits<double>::quiet_NaN(); }
            const std::vector<double>* y2Ptr = nullptr;
            if (!tr.rhsTrace.empty()) {
                y2Ptr = &get_wave(tr.rhsTrace, false);
                if (y2Ptr->size() != y1.size()) { return std::numeric_limits<double>::quiet_NaN(); }
            }

            std::vector<double> crossings;
            for (size_t i = 1; i < y1.size(); ++i) {
                if (tGrid[i] < tr.td - 1e-18) { continue; }
                const double r0     = y2Ptr ? (*y2Ptr)[i - 1] : tr.val;
                const double r1     = y2Ptr ? (*y2Ptr)[i] : tr.val;
                const double d0     = y1[i - 1] - r0;
                const double d1     = y1[i] - r1;
                const bool   isRise = (d0 < 0.0 && d1 >= 0.0) || (i == 1 && d0 == 0.0 && d1 > 0.0);
                const bool   isFall = (d0 > 0.0 && d1 <= 0.0) || (i == 1 && d0 == 0.0 && d1 < 0.0);
                bool         match  = false;
                if (tr.dir == MeasCrossDir::RISE) {
                    match = isRise;
                } else if (tr.dir == MeasCrossDir::FALL) {
                    match = isFall;
                } else {
                    match = isRise || isFall;
                }
                if (match) {
                    const double denom  = d1 - d0;
                    const double frac   = (std::abs(denom) > 0.0) ? std::clamp(-d0 / denom, 0.0, 1.0) : 0.0;
                    const double tCross = tGrid[i - 1] + frac * (tGrid[i] - tGrid[i - 1]);
                    if (tCross >= tr.td - 1e-18) { crossings.push_back(tCross); }
                }
            }
            if (crossings.empty()) { return std::numeric_limits<double>::quiet_NaN(); }
            if (tr.count < 0) { return crossings.back(); }
            if (static_cast<size_t>(tr.count) <= crossings.size()) {
                return crossings[static_cast<size_t>(tr.count - 1)];
            }
            return std::numeric_limits<double>::quiet_NaN();
        };

        auto trialParams = iObj.parameters;
        for (size_t mIdx = 0; mIdx < nMeas; ++mIdx) {
            const auto& ms  = iObj.measurements[mIdx];
            double      res = std::numeric_limits<double>::quiet_NaN();

            if (ms.kind == MeasKind::PARAM) {
                if (iObj.argSens && !iObj.marginTargets.empty()) {
                    try {
                        res = parse_param(ms.expr, trialParams);
                    } catch (...) {
                        res = std::numeric_limits<double>::quiet_NaN();
                    }
                } else {
                    res = parse_param(ms.expr, trialParams);
                }
            } else if (ms.kind == MeasKind::TRIG_TARG) {
                const double t1 = eval_trigger(ms.trig);
                const double t2 = eval_trigger(ms.targ);
                if (std::isfinite(t1) && std::isfinite(t2)) { res = t2 - t1; }
            } else if (ms.kind == MeasKind::WHEN) {
                res = eval_trigger(ms.trig);
            } else if (ms.kind == MeasKind::FIND_AT) {
                const auto& y = get_wave(ms.trace, false);
                res           = interp_wave_at(y, ms.atTime);
            } else if (ms.kind == MeasKind::FIND_WHEN) {
                const double tCond = eval_trigger(ms.trig);
                if (std::isfinite(tCond)) {
                    const auto& y = get_wave(ms.trace, false);
                    res           = interp_wave_at(y, tCond);
                }
            } else if (ms.kind == MeasKind::DERIV) {
                const double tEval = ms.trig.active ? eval_trigger(ms.trig) : ms.atTime;
                if (std::isfinite(tEval)) {
                    const auto& y = get_wave(ms.trace, false);
                    res           = deriv_wave_at(y, tEval);
                }
            } else if (ms.kind == MeasKind::COUNT || ms.kind == MeasKind::FREQ || ms.kind == MeasKind::VPEAK
                       || ms.kind == MeasKind::FWHM || ms.kind == MeasKind::DT_PI || ms.kind == MeasKind::AREA) {
                const int64_t pIdx = find_phase_event_idx(ms.trace);
                if (pIdx >= 0 && static_cast<size_t>(b) < trialEvents.size()
                    && static_cast<size_t>(pIdx) < trialEvents[static_cast<size_t>(b)].size()) {
                    double t1 = ms.hasFrom ? ms.tFrom : 0.0;
                    double t2 = ms.hasTo ? ms.tTo : (tran.tstop() > 0.0 ? tran.tstop() : 1e300);
                    if (ms.trig.active) { t1 = eval_trigger(ms.trig); }
                    if (ms.targ.active) { t2 = eval_trigger(ms.targ); }
                    if (std::isfinite(t1) && std::isfinite(t2)) {
                        if (t2 < t1) { std::swap(t1, t2); }
                        const auto&           evs = trialEvents[static_cast<size_t>(b)][static_cast<size_t>(pIdx)];
                        std::vector<SfqEvent> matched;
                        for (const auto& ev : evs) {
                            if (ev.time < t1 - 1e-18 || ev.time > t2 + 1e-18) { continue; }
                            if (ms.trig.dir == MeasCrossDir::RISE && ev.polarity <= 0) { continue; }
                            if (ms.trig.dir == MeasCrossDir::FALL && ev.polarity >= 0) { continue; }
                            matched.push_back(ev);
                        }
                        if (ms.kind == MeasKind::COUNT) {
                            res = static_cast<double>(matched.size());
                        } else if (ms.kind == MeasKind::FREQ) {
                            if (matched.size() >= 2) {
                                const double span = matched.back().time - matched.front().time;
                                res = (span > 0.0) ? (static_cast<double>(matched.size() - 1) / span) : 0.0;
                            } else {
                                res = 0.0;
                            }
                        } else if (!matched.empty()) {
                            auto get_metric = [&](const SfqEvent& ev) -> double {
                                if (ms.kind == MeasKind::VPEAK) { return std::abs(ev.vPeak); }
                                if (ms.kind == MeasKind::FWHM) { return ev.fwhm; }
                                if (ms.kind == MeasKind::DT_PI) { return ev.dtPi; }
                                return std::abs(ev.area);
                            };
                            if (ms.hasSfqIndex) {
                                if (ms.sfqIndex < 0) {
                                    res = get_metric(matched.back());
                                } else if (static_cast<size_t>(ms.sfqIndex) <= matched.size()) {
                                    res = get_metric(matched[static_cast<size_t>(ms.sfqIndex - 1)]);
                                }
                            } else {
                                double sum = 0.0;
                                for (const auto& ev : matched) { sum += get_metric(ev); }
                                res = sum / static_cast<double>(matched.size());
                            }
                        }
                    }
                }
            } else {
                const auto& y = get_wave(ms.trace, false);
                if (!y.empty() && y.size() == tGrid.size()) {
                    double t1 = ms.hasFrom ? ms.tFrom : tGrid.front();
                    double t2 = ms.hasTo ? ms.tTo : tGrid.back();
                    if (ms.trig.active) { t1 = eval_trigger(ms.trig); }
                    if (ms.targ.active) { t2 = eval_trigger(ms.targ); }
                    if (std::isfinite(t1) && std::isfinite(t2)) {
                        double sign = 1.0;
                        if (t2 < t1) {
                            std::swap(t1, t2);
                            sign = -1.0;
                        }
                        t1 = std::clamp(t1, tGrid.front(), tGrid.back());
                        t2 = std::clamp(t2, tGrid.front(), tGrid.back());

                        const double yStart = interp_wave_at(y, t1);
                        const double yEnd   = interp_wave_at(y, t2);
                        double       minVal = std::min(yStart, yEnd);
                        double       maxVal = std::max(yStart, yEnd);
                        double       minAt  = (yStart <= yEnd) ? t1 : t2;
                        double       maxAt  = (yStart >= yEnd) ? t1 : t2;
                        double       integ  = 0.0;
                        double       sqInt  = 0.0;

                        double prevT = t1;
                        double prevY = yStart;
                        auto   it    = std::upper_bound(tGrid.begin(), tGrid.end(), t1);
                        for (; it != tGrid.end() && *it < t2; ++it) {
                            const size_t idx = static_cast<size_t>(std::distance(tGrid.begin(), it));
                            const double curT = tGrid[idx];
                            const double curY = y[idx];
                            const double dt   = curT - prevT;
                            if (dt > 0.0) {
                                integ += 0.5 * (prevY + curY) * dt;
                                sqInt += (dt / 3.0) * (prevY * prevY + prevY * curY + curY * curY);
                            }
                            if (curY < minVal) {
                                minVal = curY;
                                minAt  = curT;
                            }
                            if (curY > maxVal) {
                                maxVal = curY;
                                maxAt  = curT;
                            }
                            prevT = curT;
                            prevY = curY;
                        }
                        const double dtEnd = t2 - prevT;
                        if (dtEnd > 0.0) {
                            integ += 0.5 * (prevY + yEnd) * dtEnd;
                            sqInt += (dtEnd / 3.0) * (prevY * prevY + prevY * yEnd + yEnd * yEnd);
                        }
                        const double span = t2 - t1;
                        if (ms.kind == MeasKind::INTEG) {
                            res = sign * integ;
                        } else if (ms.kind == MeasKind::AVG) {
                            res = (span > 0.0) ? (integ / span) : yStart;
                        } else if (ms.kind == MeasKind::RMS) {
                            res = (span > 0.0) ? std::sqrt(std::max(0.0, sqInt / span)) : std::abs(yStart);
                        } else if (ms.kind == MeasKind::MIN) {
                            res = minVal;
                        } else if (ms.kind == MeasKind::MAX) {
                            res = maxVal;
                        } else if (ms.kind == MeasKind::MIN_AT) {
                            res = minAt;
                        } else if (ms.kind == MeasKind::MAX_AT) {
                            res = maxAt;
                        } else if (ms.kind == MeasKind::PP) {
                            res = maxVal - minVal;
                        }
                    }
                }
            }

            measVals[mIdx][static_cast<size_t>(b)] = res;
            if (std::isfinite(res)) {
                Parameter p;
                p.set_value(res);
                trialParams[ParameterName(ms.name, std::nullopt)] = p;
                if (b == 0) {
                    const_cast<Input&>(iObj).parameters[ParameterName(ms.name, std::nullopt)] = p;
                }
            }
        }
    }

    measTrialValues = measVals;
    if (iObj.argSens && !iObj.marginTargets.empty()) {
        write_sensitivity(iObj, mObj, sObj);
        for (size_t mIdx = 0; mIdx < nMeas; ++mIdx) {
            const auto& ms = iObj.measurements[mIdx];
            if (ms.kind != MeasKind::PARAM) { continue; }
            const double res = parse_param(ms.expr, iObj.parameters);
            if (std::isfinite(res)) {
                for (int64_t b = 0; b < B; ++b) { measVals[mIdx][static_cast<size_t>(b)] = res; }
                Parameter p;
                p.set_expression(Misc::precise_to_string(res));
                p.set_value(res);
                const_cast<Input&>(iObj).parameters[ParameterName(ms.name, std::nullopt)] = p;
            }
        }
        measTrialValues = measVals;
    }
    measResults.clear();
    measResults.reserve(nMeas);
    for (size_t mIdx = 0; mIdx < nMeas; ++mIdx) {
        measResults.emplace_back(iObj.measurements[mIdx].name, measVals[mIdx][0]);
    }

    auto kind_str = [](MeasKind k) -> const char* {
        switch (k) {
            case MeasKind::TRIG_TARG: return "TRIG_TARG";
            case MeasKind::AVG:       return "AVG";
            case MeasKind::RMS:       return "RMS";
            case MeasKind::MIN:       return "MIN";
            case MeasKind::MAX:       return "MAX";
            case MeasKind::MIN_AT:    return "MIN_AT";
            case MeasKind::MAX_AT:    return "MAX_AT";
            case MeasKind::PP:        return "PP";
            case MeasKind::INTEG:     return "INTEG";
            case MeasKind::FIND_AT:   return "FIND_AT";
            case MeasKind::FIND_WHEN: return "FIND_WHEN";
            case MeasKind::WHEN:      return "WHEN";
            case MeasKind::DERIV:     return "DERIV";
            case MeasKind::PARAM:     return "PARAM";
            case MeasKind::COUNT:     return "COUNT";
            case MeasKind::FREQ:      return "FREQ";
            case MeasKind::VPEAK:     return "VPEAK";
            case MeasKind::FWHM:      return "FWHM";
            case MeasKind::DT_PI:     return "DT_PI";
            case MeasKind::AREA:      return "AREA";
        }
        return "MEAS";
    };

    auto unit_str = [&](const MeasSpec& ms) -> std::string {
        if (ms.kind == MeasKind::TRIG_TARG || ms.kind == MeasKind::WHEN || ms.kind == MeasKind::MIN_AT
            || ms.kind == MeasKind::MAX_AT || ms.kind == MeasKind::FWHM || ms.kind == MeasKind::DT_PI) {
            return "s";
        }
        if (ms.kind == MeasKind::COUNT) { return "pulses"; }
        if (ms.kind == MeasKind::FREQ) { return "Hz"; }
        if (ms.kind == MeasKind::VPEAK) { return "V"; }
        if (ms.kind == MeasKind::AREA) { return "V*s"; }
        if (ms.kind == MeasKind::PARAM) { return "-"; }
        char prefix = (iObj.argAnal == AnalysisType::Voltage) ? 'V' : 'P';
        if (ms.trace.size() > 2 && ms.trace[1] == '(') { prefix = ms.trace[0]; }
        if (ms.kind == MeasKind::INTEG) {
            if (prefix == 'V') { return "V*s"; }
            if (prefix == 'I') { return "C"; }
            return "rad*s";
        }
        if (ms.kind == MeasKind::DERIV) {
            if (prefix == 'V') { return "V/s"; }
            if (prefix == 'I') { return "A/s"; }
            return "rad/s";
        }
        if (prefix == 'V') { return "V"; }
        if (prefix == 'I') { return "A"; }
        return "rad";
    };

    if (!iObj.argMin) {
        std::cout << "\n========================================================================================================\n";
        std::cout << "JoSIM Measurement Report (" << nMeas << " measurement(s)";
        if (B > 1) { std::cout << ", " << B << " batched trials"; }
        std::cout << ")\n";
        std::cout << "========================================================================================================\n";
        if (B == 1) {
            std::cout << std::left << std::setw(20) << "Name" << std::setw(12) << "Type" << std::right << std::setw(16)
                      << "Value" << "  " << std::left << std::setw(8) << "Unit" << "Details\n";
            std::cout << "--------------------------------------------------------------------------------------------------------\n";
            for (size_t mIdx = 0; mIdx < nMeas; ++mIdx) {
                const auto&       ms   = iObj.measurements[mIdx];
                const double      val  = measVals[mIdx][0];
                const std::string unit = unit_str(ms);
                std::cout << std::left << std::setw(20) << ms.name << std::setw(12) << kind_str(ms.kind);
                if (std::isfinite(val)) {
                    std::cout << std::right << std::scientific << std::setprecision(4) << std::setw(16) << val;
                } else {
                    std::cout << std::right << std::setw(16) << "FAILED";
                }
                std::cout << "  " << std::left << std::setw(8) << unit;
                std::ostringstream det;
                if (std::isfinite(val)) {
                    if (unit == "s") {
                        det << "(" << std::fixed << std::setprecision(2) << (val * 1e12) << " ps) ";
                    } else if (unit == "V*s") {
                        det << "(" << std::fixed << std::setprecision(3) << (val / Constants::PHI_ZERO) << " Phi0) ";
                    } else if (unit == "Hz") {
                        det << "(" << std::fixed << std::setprecision(2) << (val * 1e-9) << " GHz) ";
                    }
                }
                if (ms.kind == MeasKind::PARAM) {
                    det << "EXPR=" << ms.expr;
                } else if (!ms.trace.empty()) {
                    det << ms.trace;
                } else if (!ms.trig.trace.empty()) {
                    det << ms.trig.trace << " -> " << ms.targ.trace;
                }
                std::cout << det.str() << "\n";
            }
        } else {
            std::cout << std::left << std::setw(18) << "Name" << std::setw(10) << "Type" << std::right << std::setw(14)
                      << "Mean" << std::setw(14) << "StdDev" << std::setw(14) << "Min" << std::setw(14) << "Max"
                      << std::setw(10) << "Valid" << "  " << std::left << "Unit\n";
            std::cout << "--------------------------------------------------------------------------------------------------------\n";
            for (size_t mIdx = 0; mIdx < nMeas; ++mIdx) {
                const auto& ms   = iObj.measurements[mIdx];
                double      sum  = 0.0, sumSq = 0.0, vMin = 1e300, vMax = -1e300;
                int64_t     cnt  = 0;
                for (int64_t b = 0; b < B; ++b) {
                    const double v = measVals[mIdx][static_cast<size_t>(b)];
                    if (std::isfinite(v)) {
                        sum   += v;
                        sumSq += v * v;
                        vMin   = std::min(vMin, v);
                        vMax   = std::max(vMax, v);
                        ++cnt;
                    }
                }
                std::cout << std::left << std::setw(18) << ms.name << std::setw(10) << kind_str(ms.kind);
                if (cnt > 0) {
                    const double mean = sum / static_cast<double>(cnt);
                    const double var  = std::max(0.0, (sumSq / static_cast<double>(cnt)) - mean * mean);
                    std::cout << std::right << std::scientific << std::setprecision(4) << std::setw(14) << mean
                              << std::setw(14) << std::sqrt(var) << std::setw(14) << vMin << std::setw(14) << vMax;
                } else {
                    std::cout << std::right << std::setw(14) << "FAILED" << std::setw(14) << "-" << std::setw(14) << "-"
                              << std::setw(14) << "-";
                }
                std::ostringstream vStr;
                vStr << cnt << "/" << B;
                std::cout << std::setw(10) << vStr.str() << "  " << std::left << unit_str(ms) << "\n";
            }
        }
        std::cout << "========================================================================================================\n";
    }

    if (iObj.measFile.has_value() && !iObj.measFile->empty()) {
        std::ofstream ofs(*iObj.measFile);
        if (ofs.is_open()) {
            ofs << "name,type,unit,value,mean,stddev,min,max,valid_trials,total_trials\n";
            ofs << std::scientific << std::setprecision(12);
            for (size_t mIdx = 0; mIdx < nMeas; ++mIdx) {
                const auto& ms  = iObj.measurements[mIdx];
                double      sum = 0.0, sumSq = 0.0, vMin = 1e300, vMax = -1e300;
                int64_t     cnt = 0;
                for (int64_t b = 0; b < B; ++b) {
                    const double v = measVals[mIdx][static_cast<size_t>(b)];
                    if (std::isfinite(v)) {
                        sum   += v;
                        sumSq += v * v;
                        vMin   = std::min(vMin, v);
                        vMax   = std::max(vMax, v);
                        ++cnt;
                    }
                }
                const double nom  = measVals[mIdx][0];
                const double mean = (cnt > 0) ? (sum / static_cast<double>(cnt)) : std::numeric_limits<double>::quiet_NaN();
                const double stdv = (cnt > 0) ? std::sqrt(std::max(0.0, (sumSq / static_cast<double>(cnt)) - mean * mean))
                                              : std::numeric_limits<double>::quiet_NaN();
                ofs << ms.name << "," << kind_str(ms.kind) << "," << unit_str(ms) << "," << nom << "," << mean << ","
                    << stdv << "," << (cnt > 0 ? vMin : nom) << "," << (cnt > 0 ? vMax : nom) << "," << cnt << "," << B
                    << "\n";
            }
        }
    }

    int64_t goalFailures = 0;
    for (size_t mIdx = 0; mIdx < nMeas; ++mIdx) {
        const auto& ms = iObj.measurements[mIdx];
        if (!ms.hasMinGoal && !ms.hasMaxGoal && !ms.hasValGoal) { continue; }
        const double val = measVals[mIdx][0];
        bool         ok  = std::isfinite(val);
        if (ok && ms.hasValGoal) {
            if (std::abs(val - ms.valGoal) > ms.goalTol) { ok = false; }
        }
        if (ok && ms.hasMinGoal) {
            if (val < ms.minGoal - ms.goalTol) { ok = false; }
        }
        if (ok && ms.hasMaxGoal) {
            if (val > ms.maxGoal + ms.goalTol) { ok = false; }
        }
        if (!ok) {
            ++goalFailures;
            std::cerr << "E: Measurement goal failed for '" << ms.name << "' (measured " << val << ")\n";
        }
    }
    if (goalFailures > 0) {
        std::exit(EXIT_FAILURE);
    }
}

void Output::write_loop_audit(const Input& iObj, Matrix& mObj, Simulation& sObj) {
    const auto& jjs  = sObj.get_jjs();
    const auto& inds = sObj.get_inductors();
    const int32_t nVerts = static_cast<int32_t>(mObj.nm.size()) + 1;

    struct ScEdge {
        int32_t     id        = -1;
        int32_t     u         = 0;
        int32_t     v         = 0;
        int         type      = 0; // 0 = Inductor, 1 = JJ, 2 = Source (V/P)
        int32_t     devIdx    = -1;
        std::string label;
        double      L         = 0.0;
        double      Ic        = 0.0;
        double      phiOffset = 0.0;
        double      weight    = 0.0;
    };

    std::vector<ScEdge> edges;
    edges.reserve(inds.size() + jjs.size() + mObj.components.vsIndices.size() + mObj.components.psIndices.size());

    for (size_t i = 0; i < inds.size(); ++i) {
        const auto&   ind = inds[i];
        const int32_t u   = ind.indexInfo.posIndex_ ? (static_cast<int32_t>(*ind.indexInfo.posIndex_) + 1) : 0;
        const int32_t v   = ind.indexInfo.negIndex_ ? (static_cast<int32_t>(*ind.indexInfo.negIndex_) + 1) : 0;
        if (u == v || u < 0 || v < 0 || u >= nVerts || v >= nVerts) { continue; }
        ScEdge e;
        e.id        = static_cast<int32_t>(edges.size());
        e.u         = u;
        e.v         = v;
        e.type      = 0;
        e.devIdx    = static_cast<int32_t>(i);
        e.label     = ind.netlistInfo.label_;
        e.L         = std::abs(ind.netlistInfo.value_);
        e.phiOffset = ind.phiOffset_;
        e.weight    = std::max(e.L, 1e-16);
        edges.push_back(std::move(e));
    }

    for (size_t i = 0; i < jjs.size(); ++i) {
        const auto&   jj = jjs[i];
        const int32_t u  = jj.indexInfo.posIndex_ ? (static_cast<int32_t>(*jj.indexInfo.posIndex_) + 1) : 0;
        const int32_t v  = jj.indexInfo.negIndex_ ? (static_cast<int32_t>(*jj.indexInfo.negIndex_) + 1) : 0;
        if (u == v || u < 0 || v < 0 || u >= nVerts || v >= nVerts) { continue; }
        ScEdge e;
        e.id        = static_cast<int32_t>(edges.size());
        e.u         = u;
        e.v         = v;
        e.type      = 1;
        e.devIdx    = static_cast<int32_t>(i);
        e.label     = jj.netlistInfo.label_;
        e.Ic        = std::abs(jj.model_.ic());
        e.phiOffset = jj.model_.phiOff();
        e.weight    = 5e-14; // 0.05 pH tie-breaking penalty per JJ
        edges.push_back(std::move(e));
    }

    for (const auto& vsIdx : mObj.components.vsIndices) {
        const auto&   vs = std::get<VoltageSource>(mObj.components.devices[vsIdx]);
        const int32_t u  = vs.indexInfo.posIndex_ ? (static_cast<int32_t>(*vs.indexInfo.posIndex_) + 1) : 0;
        const int32_t v  = vs.indexInfo.negIndex_ ? (static_cast<int32_t>(*vs.indexInfo.negIndex_) + 1) : 0;
        if (u == v || u < 0 || v < 0 || u >= nVerts || v >= nVerts) { continue; }
        ScEdge e;
        e.id     = static_cast<int32_t>(edges.size());
        e.u      = u;
        e.v      = v;
        e.type   = 2;
        e.label  = vs.netlistInfo.label_;
        e.weight = 1e-16;
        edges.push_back(std::move(e));
    }

    for (const auto& psIdx : mObj.components.psIndices) {
        const auto&   ps = std::get<PhaseSource>(mObj.components.devices[psIdx]);
        const int32_t u  = ps.indexInfo.posIndex_ ? (static_cast<int32_t>(*ps.indexInfo.posIndex_) + 1) : 0;
        const int32_t v  = ps.indexInfo.negIndex_ ? (static_cast<int32_t>(*ps.indexInfo.negIndex_) + 1) : 0;
        if (u == v || u < 0 || v < 0 || u >= nVerts || v >= nVerts) { continue; }
        ScEdge e;
        e.id     = static_cast<int32_t>(edges.size());
        e.u      = u;
        e.v      = v;
        e.type   = 2;
        e.label  = ps.netlistInfo.label_;
        e.weight = 1e-16;
        edges.push_back(std::move(e));
    }

    std::vector<std::vector<int32_t>> adj(static_cast<size_t>(nVerts));
    for (const auto& e : edges) {
        adj[static_cast<size_t>(e.u)].push_back(e.id);
        adj[static_cast<size_t>(e.v)].push_back(e.id);
    }

    struct AuditedLoop {
        std::vector<int32_t>     canonicalKey;
        std::vector<std::string> jjLabels;
        std::vector<std::string> indLabels;
        double                   Ltot     = 0.0;
        double                   IcMin    = 0.0;
        double                   IcMax    = 0.0;
        double                   betaL    = 0.0;
        double                   fluxPhi0 = 0.0;
        std::string              classification;
    };

    std::vector<AuditedLoop> loops;
    std::vector<double>      dist(static_cast<size_t>(nVerts), 1e300);
    std::vector<int32_t>     parentVert(static_cast<size_t>(nVerts), -1);
    std::vector<int32_t>     parentEdge(static_cast<size_t>(nVerts), -1);
    std::vector<uint8_t>     visited(static_cast<size_t>(nVerts), 0);

    struct HeapEntry {
        double  d;
        int32_t v;
        bool    operator<(const HeapEntry& o) const { return d > o.d; } // min-heap
    };
    std::vector<HeapEntry> heap;

    for (const auto& seedEdge : edges) {
        if (seedEdge.type != 0 && seedEdge.type != 1) { continue; }

        std::fill(dist.begin(), dist.end(), 1e300);
        std::fill(parentVert.begin(), parentVert.end(), -1);
        std::fill(parentEdge.begin(), parentEdge.end(), -1);
        std::fill(visited.begin(), visited.end(), 0);
        heap.clear();

        const int32_t src = seedEdge.v;
        const int32_t dst = seedEdge.u;
        dist[static_cast<size_t>(src)] = 0.0;
        heap.push_back({0.0, src});

        while (!heap.empty()) {
            std::pop_heap(heap.begin(), heap.end());
            const HeapEntry top = heap.back();
            heap.pop_back();
            const int32_t x = top.v;
            if (visited[static_cast<size_t>(x)]) { continue; }
            visited[static_cast<size_t>(x)] = 1;
            if (x == dst) { break; }

            for (int32_t eId : adj[static_cast<size_t>(x)]) {
                if (eId == seedEdge.id) { continue; }
                const auto&   ce = edges[static_cast<size_t>(eId)];
                const int32_t y  = (ce.u == x) ? ce.v : ce.u;
                if (visited[static_cast<size_t>(y)]) { continue; }
                const double nd = top.d + ce.weight;
                if (nd < dist[static_cast<size_t>(y)]) {
                    dist[static_cast<size_t>(y)]       = nd;
                    parentVert[static_cast<size_t>(y)] = x;
                    parentEdge[static_cast<size_t>(y)] = eId;
                    heap.push_back({nd, y});
                    std::push_heap(heap.begin(), heap.end());
                }
            }
        }

        if (!visited[static_cast<size_t>(dst)]) { continue; }

        std::vector<std::pair<int32_t, double>> cycleEdges;
        cycleEdges.push_back({seedEdge.id, 1.0}); // traversed seedEdge.u -> seedEdge.v
        int32_t cur = dst;
        while (cur != src) {
            const int32_t eId  = parentEdge[static_cast<size_t>(cur)];
            const int32_t prev = parentVert[static_cast<size_t>(cur)];
            if (eId < 0 || prev < 0) { break; }
            const auto&  ce   = edges[static_cast<size_t>(eId)];
            const double sign = (ce.u == prev && ce.v == cur) ? 1.0 : -1.0;
            cycleEdges.push_back({eId, sign});
            cur = prev;
        }

        std::vector<int32_t> key;
        AuditedLoop          al;
        double               icMin   = 1e300;
        double               icMax   = 0.0;
        double               phiSum  = 0.0;
        for (const auto& es : cycleEdges) {
            const auto& ce = edges[static_cast<size_t>(es.first)];
            if (ce.type == 0 || ce.type == 1) {
                key.push_back(ce.id);
            }
            if (ce.type == 0) {
                al.indLabels.push_back(ce.label);
                al.Ltot += ce.L;
                phiSum  += es.second * ce.phiOffset;
            } else if (ce.type == 1) {
                al.jjLabels.push_back(ce.label);
                icMin  = std::min(icMin, ce.Ic);
                icMax  = std::max(icMax, ce.Ic);
                phiSum += es.second * ce.phiOffset;
            }
        }
        if (al.jjLabels.empty()) { continue; }

        std::sort(key.begin(), key.end());
        key.erase(std::unique(key.begin(), key.end()), key.end());
        bool dup = false;
        for (const auto& existing : loops) {
            if (existing.canonicalKey == key) {
                dup = true;
                break;
            }
        }
        if (dup) { continue; }

        std::sort(al.jjLabels.begin(), al.jjLabels.end());
        std::sort(al.indLabels.begin(), al.indLabels.end());
        al.canonicalKey = std::move(key);
        al.IcMin        = (icMin < 1e290) ? icMin : 0.0;
        al.IcMax        = icMax;
        al.betaL        = (al.Ltot * al.IcMin) / Constants::SIGMA;
        al.fluxPhi0     = std::abs(phiSum) / (2.0 * Constants::PI);
        if (al.Ltot <= 1e-18) {
            al.classification = "DEGENERATE_0L";
        } else if (al.betaL < 0.1 * Constants::PI) {
            al.classification = "SUB_CRITICAL";
        } else if (al.betaL < Constants::PI) {
            al.classification = "NON_STORAGE";
        } else if (al.betaL < 3.0 * Constants::PI) {
            al.classification = "STORAGE_QUANT";
        } else {
            al.classification = "MULTI_FLUX_TRAP";
        }
        loops.push_back(std::move(al));
    }

    auto& mutParams = const_cast<Input&>(iObj).parameters;
    auto  set_param = [&](const std::string& pName, double val) {
        Parameter p;
        p.set_expression(Misc::precise_to_string(val));
        p.set_value(val);
        mutParams[ParameterName(pName, std::nullopt)] = p;
    };

    int64_t degCount  = 0;
    int64_t storCount = 0;
    std::unordered_map<std::string, size_t> jjBestLoop;
    for (size_t i = 0; i < loops.size(); ++i) {
        const auto& lp = loops[i];
        if (lp.classification == "DEGENERATE_0L") { ++degCount; }
        if (lp.betaL >= Constants::PI) { ++storCount; }
        const std::string idxStr = std::to_string(i + 1);
        set_param("BETA_L_LOOP_" + idxStr, lp.betaL);
        set_param("L_LOOP_" + idxStr, lp.Ltot);
        set_param("FLUX_LOOP_" + idxStr, lp.fluxPhi0);
        for (const auto& jl : lp.jjLabels) {
            auto it = jjBestLoop.find(jl);
            if (it == jjBestLoop.end() || lp.Ltot < loops[it->second].Ltot) {
                jjBestLoop[jl] = i;
                set_param("BETA_L_" + jl, lp.betaL);
                set_param("L_LOOP_" + jl, lp.Ltot);
                set_param("FLUX_LOOP_" + jl, lp.fluxPhi0);
            }
        }
    }
    set_param("LOOP_COUNT", static_cast<double>(loops.size()));
    set_param("LOOP_DEGENERATE_COUNT", static_cast<double>(degCount));
    set_param("LOOP_STORAGE_COUNT", static_cast<double>(storCount));

    auto join_list = [](const std::vector<std::string>& v, size_t maxLen) -> std::string {
        if (v.empty()) { return "-"; }
        std::string s = v.front();
        for (size_t i = 1; i < v.size(); ++i) { s += "+" + v[i]; }
        if (s.size() > maxLen && maxLen > 3) { s = s.substr(0, maxLen - 3) + "..."; }
        return s;
    };

    std::cout << "\n==============================================================================================================\n";
    std::cout << "JoSIM Superconducting Loop (beta_L) Audit Report (" << loops.size() << " loops, " << storCount
              << " storage, " << degCount << " degenerate)\n";
    std::cout << "--------------------------------------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(8) << "Loop" << std::setw(22) << "Junctions" << std::setw(22) << "Inductors"
              << std::right << std::setw(12) << "L_loop(pH)" << std::setw(12) << "Ic_min(mA)" << std::setw(10)
              << "beta_L" << std::setw(11) << "Flux(Phi0)" << "  " << std::left << "Class\n";
    std::cout << "--------------------------------------------------------------------------------------------------------------\n";
    for (size_t i = 0; i < loops.size(); ++i) {
        const auto& lp = loops[i];
        std::cout << std::left << std::setw(8) << ("#" + std::to_string(i + 1)) << std::setw(22)
                  << join_list(lp.jjLabels, 20) << std::setw(22) << join_list(lp.indLabels, 20) << std::right
                  << std::fixed << std::setprecision(3) << std::setw(12) << (lp.Ltot * 1e12) << std::setw(12)
                  << (lp.IcMin * 1e3) << std::setw(10) << lp.betaL << std::setw(11) << lp.fluxPhi0 << "  " << std::left
                  << lp.classification << "\n";
    }
    std::cout << "==============================================================================================================\n";

    if (iObj.loopFile.has_value() && !iObj.loopFile->empty()) {
        std::ofstream ofs(*iObj.loopFile);
        if (ofs.is_open()) {
            ofs << "loop_id,junctions,inductors,L_loop_H,Ic_min_A,Ic_max_A,beta_L,flux_Phi0,classification\n";
            ofs << std::scientific << std::setprecision(12);
            for (size_t i = 0; i < loops.size(); ++i) {
                const auto& lp = loops[i];
                ofs << (i + 1) << ",\"" << join_list(lp.jjLabels, 256) << "\",\"" << join_list(lp.indLabels, 256)
                    << "\"," << lp.Ltot << "," << lp.IcMin << "," << lp.IcMax << "," << lp.betaL << "," << lp.fluxPhi0
                    << "," << lp.classification << "\n";
            }
        }
    }

    if (iObj.argCheckLoopsStrict && degCount > 0) {
        std::cerr << "E: Superconducting loop audit failed: " << degCount
                  << " degenerate zero-inductance JJ loop(s) detected in STRICT mode.\n";
        std::exit(EXIT_FAILURE);
    }
}

void Output::evaluate_jitter(const Input& iObj, Matrix& mObj, Simulation& sObj) {
    if (iObj.jitterSpecs.empty()) { return; }

    std::vector<PhaseTraceInfo>                     phaseTraces;
    std::vector<std::vector<std::vector<SfqEvent>>> trialEvents;
    extract_trial_phase_events(iObj, mObj, sObj, phaseTraces, trialEvents);
    if (trialEvents.empty()) { return; }

    auto glob_match = [](const std::string& text, const std::string& pat) -> bool {
        size_t ti = 0, pi = 0, starIdx = std::string::npos, matchIdx = 0;
        while (ti < text.size()) {
            if (pi < pat.size() && (pat[pi] == '?' || pat[pi] == text[ti])) {
                ++ti;
                ++pi;
            } else if (pi < pat.size() && pat[pi] == '*') {
                starIdx  = pi++;
                matchIdx = ti;
            } else if (starIdx != std::string::npos) {
                pi = starIdx + 1;
                ti = ++matchIdx;
            } else {
                return false;
            }
        }
        while (pi < pat.size() && pat[pi] == '*') { ++pi; }
        return pi == pat.size();
    };

    auto find_trace_indices = [&](std::string rawName) -> std::vector<int64_t> {
        std::vector<int64_t> out;
        std::string          norm = normalize_event_trace_label(rawName);
        std::replace(norm.begin(), norm.end(), '.', '|');
        std::string inner = norm;
        if (inner.size() > 3 && (inner.rfind("P(", 0) == 0 || inner.rfind("V(", 0) == 0) && inner.back() == ')') {
            inner = inner.substr(2, inner.size() - 3);
        }
        const std::string withP   = "P(" + inner + ")";
        const bool        hasGlob = (inner.find('*') != std::string::npos || inner.find('?') != std::string::npos);
        for (size_t k = 0; k < phaseTraces.size(); ++k) {
            std::string ptNorm = normalize_event_trace_label(phaseTraces[k].label);
            std::replace(ptNorm.begin(), ptNorm.end(), '.', '|');
            std::string ptInner = ptNorm;
            if (ptInner.size() > 3 && ptInner.rfind("P(", 0) == 0 && ptInner.back() == ')') {
                ptInner = ptInner.substr(2, ptInner.size() - 3);
            }
            if (!hasGlob) {
                if (ptNorm == withP || ptNorm == inner) {
                    out.push_back(static_cast<int64_t>(k));
                    break;
                }
            } else {
                if (glob_match(ptInner, inner) || glob_match(ptNorm, withP)) {
                    out.push_back(static_cast<int64_t>(k));
                }
            }
        }
        return out;
    };

    auto sanitize_ident = [](std::string s) -> std::string {
        if (s.size() > 3 && s.rfind("P(", 0) == 0 && s.back() == ')') {
            s = s.substr(2, s.size() - 3);
        }
        std::string out;
        out.reserve(s.size());
        for (char c : s) {
            if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
                out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
            } else {
                out.push_back('_');
            }
        }
        return out;
    };

    struct JitterRow {
        std::string label;
        std::string refDesc;
        double      meanLatency = 0.0;
        double      rmsJitter   = 0.0;
        double      ppJitter    = 0.0;
        double      c2cJitter   = 0.0;
        int64_t     expected    = 0;
        int64_t     matched     = 0;
        int64_t     missed      = 0;
        int64_t     spurious    = 0;
        int64_t     errors      = 0;
        double      ber         = 0.0;
        bool        passed      = true;
        bool        strict      = false;
        std::string failReason;
    };

    std::vector<JitterRow> rows;
    const size_t           nTrials  = (iObj.argMargins || iObj.argSens) ? 1 : trialEvents.size();
    auto&                  paramMap = const_cast<Input&>(iObj).parameters;
    auto                   set_param = [&](const std::string& name, double val) {
        Parameter p;
        p.set_expression(Misc::precise_to_string(val));
        p.set_value(val);
        paramMap[ParameterName(name, std::nullopt)] = p;
    };

    for (const auto& js : iObj.jitterSpecs) {
        const std::vector<int64_t> tIndices = find_trace_indices(js.trace);
        for (const int64_t tIdx : tIndices) {
            JitterRow   row;
            std::string rawLbl = normalize_event_trace_label(phaseTraces[static_cast<size_t>(tIdx)].label);
            if (rawLbl.size() > 3 && rawLbl.rfind("P(", 0) == 0 && rawLbl.back() == ')') {
                rawLbl = rawLbl.substr(2, rawLbl.size() - 3);
            }
            row.label  = (!js.name.empty() && tIndices.size() == 1) ? (js.name + "(" + rawLbl + ")") : rawLbl;
            row.strict = js.strict;

            const double tFrom = js.hasFrom ? js.tFrom : -1e30;
            const double tTo   = js.hasTo ? js.tTo : 1e30;
            auto         filter_evs = [&](const std::vector<SfqEvent>& src) -> std::vector<SfqEvent> {
                std::vector<SfqEvent> out;
                for (const auto& e : src) {
                    if (e.time >= tFrom - 1e-15 && e.time <= tTo + 1e-15) { out.push_back(e); }
                }
                return out;
            };

            std::vector<double> dtSamples;
            std::vector<double> c2cDiffs;
            bool                useZeroMeanResiduals = false;
            double              explicitMeanLatency  = 0.0;

            if (js.mode == JitterMode::REF_TRACE || !js.refTrace.empty()) {
                const auto    rIndices = find_trace_indices(js.refTrace);
                const int64_t rIdx     = rIndices.empty() ? -1 : rIndices.front();
                std::string   refLbl   = js.refTrace;
                if (rIdx >= 0) {
                    refLbl = normalize_event_trace_label(phaseTraces[static_cast<size_t>(rIdx)].label);
                    if (refLbl.size() > 3 && refLbl.rfind("P(", 0) == 0 && refLbl.back() == ')') {
                        refLbl = refLbl.substr(2, refLbl.size() - 3);
                    }
                }
                row.refDesc = "REF:" + refLbl;
                if (rIdx >= 0) {
                    for (size_t b = 0; b < nTrials; ++b) {
                        const auto        refEvs  = filter_evs(trialEvents[b][static_cast<size_t>(rIdx)]);
                        const auto        targEvs = filter_evs(trialEvents[b][static_cast<size_t>(tIdx)]);
                        std::vector<bool> claimed(targEvs.size(), false);
                        bool              hasPrevDt = false;
                        double            prevDt    = 0.0;
                        double            estDelay  = js.hasNomDelay ? js.nomDelay : 0.0;
                        bool              hasEst    = js.hasNomDelay;
                        row.expected += (js.expCount >= 0) ? js.expCount : static_cast<int64_t>(refEvs.size());

                        for (size_t r = 0; r < refEvs.size(); ++r) {
                            const double tRef = refEvs[r].time;
                            if (js.hasTd && tRef < js.td - 1e-15) { continue; }
                            const double tMin = js.hasNomDelay
                                                        ? (tRef + js.nomDelay - (js.maxDelay > 0.0 ? js.maxDelay : 1e30))
                                                        : (tRef - 1e-15);
                            double       tMax = 1e30;
                            if (js.maxDelay > 0.0) {
                                tMax = tRef + (js.hasNomDelay ? js.nomDelay : 0.0) + js.maxDelay;
                            } else if (r + 1 < refEvs.size()) {
                                const double refGap = refEvs[r + 1].time - tRef;
                                tMax                = hasEst ? (tRef + estDelay + 0.5 * refGap) : (tRef + 0.9 * refGap);
                            }
                            int64_t bestM    = -1;
                            double  bestDist = 1e30;
                            for (size_t m = 0; m < targEvs.size(); ++m) {
                                if (claimed[m] || targEvs[m].polarity != refEvs[r].polarity) { continue; }
                                if (targEvs[m].time < tMin || targEvs[m].time > tMax) { continue; }
                                const double dt   = targEvs[m].time - tRef;
                                const double dist = hasEst ? std::abs(dt - estDelay) : dt;
                                if (dist < bestDist) {
                                    bestDist = dist;
                                    bestM    = static_cast<int64_t>(m);
                                    if (!hasEst) { break; }
                                }
                            }
                            if (bestM >= 0) {
                                claimed[static_cast<size_t>(bestM)] = true;
                                const double rawDt = targEvs[static_cast<size_t>(bestM)].time - tRef;
                                if (!js.hasNomDelay) {
                                    estDelay = hasEst ? (0.5 * estDelay + 0.5 * rawDt) : rawDt;
                                    hasEst   = true;
                                }
                                const double dt = rawDt - (js.hasNomDelay ? js.nomDelay : 0.0);
                                dtSamples.push_back(dt);
                                ++row.matched;
                                if (hasPrevDt) { c2cDiffs.push_back(dt - prevDt); }
                                prevDt    = dt;
                                hasPrevDt = true;
                            } else {
                                ++row.missed;
                                hasPrevDt = false;
                            }
                        }
                        for (size_t m = 0; m < targEvs.size(); ++m) {
                            if (!claimed[m] && (!js.hasTd || targEvs[m].time >= js.td - 1e-15)) { ++row.spurious; }
                        }
                    }
                }
            } else if (js.mode == JitterMode::CLOCK_GRID || js.per > 0.0) {
                std::ostringstream rDesc;
                rDesc << "CLK:" << std::fixed << std::setprecision(1) << (js.per * 1e12) << "ps";
                if (!js.bits.empty()) { rDesc << "(BITS)"; }
                row.refDesc = rDesc.str();

                const auto trial0Evs = filter_evs(trialEvents[0][static_cast<size_t>(tIdx)]);
                for (size_t b = 0; b < nTrials; ++b) {
                    const auto        targEvs = filter_evs(trialEvents[b][static_cast<size_t>(tIdx)]);
                    std::vector<bool> claimed(targEvs.size(), false);
                    bool              hasPrevDt = false;
                    double            prevDt    = 0.0;

                    if (!js.bits.empty()) {
                        const double win   = (js.window > 0.0) ? js.window : js.per;
                        const size_t nBits = js.bits.size();
                        for (size_t k = 0; k < nBits; ++k) {
                            const double      tSlot = js.td + static_cast<double>(k) * js.per;
                            std::vector<size_t> slotEvs;
                            for (size_t m = 0; m < targEvs.size(); ++m) {
                                if (!claimed[m] && targEvs[m].time >= tSlot - 1e-15
                                    && targEvs[m].time < tSlot + win - 1e-15) {
                                    slotEvs.push_back(m);
                                }
                            }
                            const char bit = js.bits[k];
                            if (bit == '1') {
                                ++row.expected;
                                if (!slotEvs.empty()) {
                                    const size_t m0 = slotEvs.front();
                                    claimed[m0]     = true;
                                    const double dt = (targEvs[m0].time - tSlot) - (js.hasNomDelay ? js.nomDelay : 0.0);
                                    dtSamples.push_back(dt);
                                    ++row.matched;
                                    if (hasPrevDt) { c2cDiffs.push_back(dt - prevDt); }
                                    prevDt    = dt;
                                    hasPrevDt = true;
                                    for (size_t s = 1; s < slotEvs.size(); ++s) {
                                        claimed[slotEvs[s]] = true;
                                        ++row.spurious;
                                    }
                                } else {
                                    ++row.missed;
                                    hasPrevDt = false;
                                }
                            } else if (bit == '0') {
                                ++row.expected;
                                for (size_t m : slotEvs) {
                                    claimed[m] = true;
                                    ++row.spurious;
                                }
                            } else {
                                for (size_t m : slotEvs) { claimed[m] = true; }
                            }
                        }
                        for (size_t m = 0; m < targEvs.size(); ++m) {
                            if (!claimed[m] && targEvs[m].time >= js.td - 1e-15) { ++row.spurious; }
                        }
                    } else {
                        double t0 = js.hasTd ? (js.td + (js.hasNomDelay ? js.nomDelay : 0.0))
                                             : (!trial0Evs.empty() ? trial0Evs.front().time
                                                                   : (!targEvs.empty() ? targEvs.front().time : 0.0));
                        if (js.sparse) {
                            row.expected += (js.expCount >= 0) ? js.expCount : static_cast<int64_t>(targEvs.size());
                            for (const auto& e : targEvs) {
                                if (js.hasTd && e.time < js.td - 1e-15) { continue; }
                                const int64_t cyc = (js.hasTd && !js.hasNomDelay)
                                                            ? static_cast<int64_t>(std::floor((e.time - js.td + 1e-15) / js.per))
                                                            : static_cast<int64_t>(std::llround((e.time - t0) / js.per));
                                const double  tGrid = (js.hasTd && !js.hasNomDelay)
                                                              ? (js.td + static_cast<double>(cyc) * js.per)
                                                              : (t0 + static_cast<double>(cyc) * js.per);
                                const double  dt    = e.time - tGrid;
                                dtSamples.push_back(dt);
                                ++row.matched;
                                if (hasPrevDt) { c2cDiffs.push_back(dt - prevDt); }
                                prevDt    = dt;
                                hasPrevDt = true;
                            }
                        } else {
                            const int64_t nCyc = (js.expCount >= 0)
                                                         ? js.expCount
                                                         : static_cast<int64_t>(!trial0Evs.empty() ? trial0Evs.size()
                                                                                                   : targEvs.size());
                            const double  win  = (js.window > 0.0) ? js.window : js.per;
                            for (int64_t k = 0; k < nCyc; ++k) {
                                ++row.expected;
                                double tRef   = 0.0;
                                double slotLo = 0.0;
                                double slotHi = 0.0;
                                if (js.hasTd && !js.hasNomDelay) {
                                    tRef   = js.td + static_cast<double>(k) * js.per;
                                    slotLo = tRef;
                                    slotHi = tRef + win;
                                } else {
                                    tRef   = t0 + static_cast<double>(k) * js.per;
                                    slotLo = tRef - 0.5 * win;
                                    slotHi = tRef + 0.5 * win;
                                }
                                std::vector<size_t> slotEvs;
                                for (size_t m = 0; m < targEvs.size(); ++m) {
                                    if (!claimed[m] && targEvs[m].time >= slotLo - 1e-15
                                        && targEvs[m].time < slotHi - 1e-15) {
                                        slotEvs.push_back(m);
                                    }
                                }
                                if (!slotEvs.empty()) {
                                    const size_t m0 = slotEvs.front();
                                    claimed[m0]     = true;
                                    const double dt = targEvs[m0].time - tRef;
                                    dtSamples.push_back(dt);
                                    ++row.matched;
                                    if (hasPrevDt) { c2cDiffs.push_back(dt - prevDt); }
                                    prevDt    = dt;
                                    hasPrevDt = true;
                                    for (size_t s = 1; s < slotEvs.size(); ++s) {
                                        claimed[slotEvs[s]] = true;
                                        ++row.spurious;
                                    }
                                } else {
                                    ++row.missed;
                                    hasPrevDt = false;
                                }
                            }
                            for (size_t m = 0; m < targEvs.size(); ++m) {
                                if (!claimed[m] && (!js.hasTd || targEvs[m].time >= js.td - 1e-15)) { ++row.spurious; }
                            }
                        }
                    }
                }
            } else {
                // ENSEMBLE or single-run period jitter mode
                const auto trial0Evs = filter_evs(trialEvents[0][static_cast<size_t>(tIdx)]);
                if (nTrials > 1) {
                    row.refDesc          = "ENSEMBLE(" + std::to_string(nTrials) + ")";
                    useZeroMeanResiduals = true;
                    const int64_t K      = (js.expCount >= 0) ? js.expCount : static_cast<int64_t>(trial0Evs.size());
                    std::vector<double>  meanT(static_cast<size_t>( std::max<int64_t>(0, K)), 0.0);
                    std::vector<int64_t> cntT(static_cast<size_t>(std::max<int64_t>(0, K)), 0);
                    for (size_t b = 0; b < nTrials; ++b) {
                        const auto targEvs = filter_evs(trialEvents[b][static_cast<size_t>(tIdx)]);
                        for (int64_t k = 0; k < K && k < static_cast<int64_t>(targEvs.size()); ++k) {
                            meanT[static_cast<size_t>(k)] += targEvs[static_cast<size_t>(k)].time;
                            ++cntT[static_cast<size_t>(k)];
                        }
                    }
                    for (int64_t k = 0; k < K; ++k) {
                        if (cntT[static_cast<size_t>(k)] > 0) {
                            meanT[static_cast<size_t>(k)] /= static_cast<double>(cntT[static_cast<size_t>(k)]);
                        }
                    }
                    if (K > 0) { explicitMeanLatency = meanT[0]; }
                    for (size_t b = 0; b < nTrials; ++b) {
                        const auto   targEvs = filter_evs(trialEvents[b][static_cast<size_t>(tIdx)]);
                        const int64_t nEv    = static_cast<int64_t>(targEvs.size());
                        row.expected        += K;
                        row.matched         += std::min<int64_t>(K, nEv);
                        if (nEv < K) { row.missed += (K - nEv); }
                        if (nEv > K) { row.spurious += (nEv - K); }
                        bool   hasPrev = false;
                        double prevRes = 0.0;
                        for (int64_t k = 0; k < std::min<int64_t>(K, nEv); ++k) {
                            const double res = targEvs[static_cast<size_t>(k)].time - meanT[static_cast<size_t>(k)];
                            dtSamples.push_back(res);
                            if (hasPrev) { c2cDiffs.push_back(res - prevRes); }
                            prevRes = res;
                            hasPrev = true;
                        }
                    }
                } else {
                    row.refDesc        = "SELF_PERIOD";
                    const int64_t nEv  = static_cast<int64_t>(trial0Evs.size());
                    const int64_t expK = (js.expCount >= 0) ? js.expCount : nEv;
                    row.expected       = expK;
                    row.matched        = std::min<int64_t>(expK, nEv);
                    if (nEv < expK) { row.missed = expK - nEv; }
                    if (nEv > expK) { row.spurious = nEv - expK; }
                    if (nEv >= 2) {
                        for (int64_t k = 1; k < nEv; ++k) {
                            const double perK = trial0Evs[static_cast<size_t>(k)].time
                                                - trial0Evs[static_cast<size_t>(k - 1)].time;
                            dtSamples.push_back(perK);
                            if (k >= 2) {
                                c2cDiffs.push_back(perK - dtSamples[dtSamples.size() - 2]);
                            }
                        }
                    } else if (nEv == 1) {
                        explicitMeanLatency  = trial0Evs[0].time;
                        useZeroMeanResiduals = true;
                        dtSamples.push_back(0.0);
                    }
                }
            }

            if (!dtSamples.empty()) {
                double sum   = 0.0;
                double minDt = dtSamples.front();
                double maxDt = dtSamples.front();
                for (double v : dtSamples) {
                    sum   += v;
                    minDt  = std::min(minDt, v);
                    maxDt  = std::max(maxDt, v);
                }
                const double mean = sum / static_cast<double>(dtSamples.size());
                double       sq   = 0.0;
                for (double v : dtSamples) {
                    const double d = v - mean;
                    sq            += d * d;
                }
                row.meanLatency = useZeroMeanResiduals ? explicitMeanLatency : mean;
                row.rmsJitter   = std::sqrt(sq / static_cast<double>(dtSamples.size()));
                row.ppJitter    = maxDt - minDt;
            }
            if (!c2cDiffs.empty()) {
                double sq = 0.0;
                for (double d : c2cDiffs) { sq += d * d; }
                row.c2cJitter = std::sqrt(sq / static_cast<double>(c2cDiffs.size()));
            }

            row.errors = row.missed + row.spurious;
            const int64_t denom = (row.expected > 0) ? row.expected : std::max<int64_t>(1, row.matched + row.errors);
            row.ber    = static_cast<double>(row.errors) / static_cast<double>(denom);

            if (js.hasMaxRms && row.rmsJitter > js.maxRms + 1e-18) {
                row.passed = false;
                row.failReason = "RMS>" + Misc::precise_to_string(js.maxRms * 1e12) + "ps";
            }
            if (js.hasMaxPp && row.ppJitter > js.maxPp + 1e-18) {
                row.passed = false;
                row.failReason = "PP>" + Misc::precise_to_string(js.maxPp * 1e12) + "ps";
            }
            if (js.hasMaxSkew && std::abs(row.meanLatency) > js.maxSkew + 1e-18) {
                row.passed = false;
                row.failReason = "|SKEW|>" + Misc::precise_to_string(js.maxSkew * 1e12) + "ps";
            }
            if (js.hasMaxBer && row.ber > js.maxBer + 1e-15) {
                row.passed = false;
                row.failReason = "BER>" + Misc::precise_to_string(js.maxBer);
            }
            if (js.hasMaxErrs && row.errors > js.maxErrs) {
                row.passed = false;
                row.failReason = "ERRS>" + std::to_string(js.maxErrs);
            }

            std::vector<std::string> exportKeys = {sanitize_ident(rawLbl)};
            if (!js.name.empty()) {
                exportKeys.push_back(sanitize_ident(js.name));
            }
            for (const auto& k : exportKeys) {
                if (k.empty()) { continue; }
                set_param("JITTER_RMS_" + k, row.rmsJitter);
                set_param("JITTER_PP_" + k, row.ppJitter);
                set_param("JITTER_C2C_" + k, row.c2cJitter);
                set_param("LATENCY_" + k, row.meanLatency);
                set_param("SKEW_" + k, row.meanLatency);
                set_param("BER_" + k, row.ber);
                set_param("ERRORS_" + k, static_cast<double>(row.errors));
            }

            rows.push_back(std::move(row));
        }
    }

    if (rows.empty()) { return; }

    std::cout << "\n===================================================================================================================\n";
    std::cout << "JoSIM Timing Jitter, Skew & BER Report (" << rows.size() << " target(s), " << nTrials << " trial(s))\n";
    std::cout << "-------------------------------------------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(18) << "Target" << std::setw(20) << "Ref / Mode" << std::right << std::setw(13)
              << "Lat/Skew(ps)" << std::setw(13) << "RMS Jit(ps)" << std::setw(13) << "PP Jit(ps)" << std::setw(13)
              << "C2C Jit(ps)" << std::setw(12) << "Match/Exp" << std::setw(6) << "Errs" << std::setw(12) << "BER"
              << "  " << std::left << "Status\n";
    std::cout << "-------------------------------------------------------------------------------------------------------------------\n";

    bool anyStrictFail = false;
    for (const auto& r : rows) {
        const std::string matchExp = std::to_string(r.matched) + "/" + std::to_string(r.expected);
        const std::string status   = r.passed ? "PASS" : ("FAIL(" + r.failReason + ")");
        if (!r.passed && r.strict) { anyStrictFail = true; }
        std::cout << std::left << std::setw(18) << r.label.substr(0, 17) << std::setw(20) << r.refDesc.substr(0, 19)
                  << std::right << std::fixed << std::setprecision(3) << std::setw(13) << (r.meanLatency * 1e12)
                  << std::setw(13) << (r.rmsJitter * 1e12) << std::setw(13) << (r.ppJitter * 1e12) << std::setw(13)
                  << (r.c2cJitter * 1e12) << std::setw(12) << matchExp << std::setw(6) << r.errors << std::scientific
                  << std::setprecision(2) << std::setw(12) << r.ber << "  " << std::left << status << "\n";
    }
    std::cout << "===================================================================================================================\n";

    if (iObj.jitterFile.has_value() && !iObj.jitterFile->empty()) {
        std::ofstream ofs(*iObj.jitterFile);
        if (ofs.is_open()) {
            const std::string& fn = *iObj.jitterFile;
            bool isJson           = (fn.size() >= 5 && (fn.substr(fn.size() - 5) == ".json" || fn.substr(fn.size() - 5) == ".JSON"));
            if (isJson) {
                ofs << "{\n  \"trials\": " << nTrials << ",\n  \"targets\": [\n";
                ofs << std::scientific << std::setprecision(12);
                for (size_t i = 0; i < rows.size(); ++i) {
                    const auto& r = rows[i];
                    ofs << "    {\n"
                        << "      \"target\": \"" << r.label << "\",\n"
                        << "      \"ref_mode\": \"" << r.refDesc << "\",\n"
                        << "      \"mean_latency_s\": " << r.meanLatency << ",\n"
                        << "      \"rms_jitter_s\": " << r.rmsJitter << ",\n"
                        << "      \"pp_jitter_s\": " << r.ppJitter << ",\n"
                        << "      \"c2c_jitter_s\": " << r.c2cJitter << ",\n"
                        << "      \"expected\": " << r.expected << ",\n"
                        << "      \"matched\": " << r.matched << ",\n"
                        << "      \"missed\": " << r.missed << ",\n"
                        << "      \"spurious\": " << r.spurious << ",\n"
                        << "      \"errors\": " << r.errors << ",\n"
                        << "      \"ber\": " << r.ber << ",\n"
                        << "      \"status\": \"" << (r.passed ? "PASS" : "FAIL") << "\"\n"
                        << "    }" << (i + 1 < rows.size() ? "," : "") << "\n";
                }
                ofs << "  ]\n}\n";
            } else {
                ofs << "target,ref_mode,mean_latency_s,rms_jitter_s,pp_jitter_s,c2c_jitter_s,expected,matched,missed,spurious,errors,ber,status\n";
                ofs << std::scientific << std::setprecision(12);
                for (const auto& r : rows) {
                    ofs << "\"" << r.label << "\",\"" << r.refDesc << "\"," << r.meanLatency << "," << r.rmsJitter << ","
                        << r.ppJitter << "," << r.c2cJitter << "," << r.expected << "," << r.matched << "," << r.missed
                        << "," << r.spurious << "," << r.errors << "," << r.ber << "," << (r.passed ? "PASS" : "FAIL")
                        << "\n";
                }
            }
        }
    }

    if (anyStrictFail) {
        std::cerr << "E: One or more STRICT .jitter / BER specifications failed.\n";
        std::exit(EXIT_FAILURE);
    }
}

void Output::write_sensitivity(const Input& iObj, Matrix& mObj, Simulation& sObj) {
    if (iObj.marginTargets.empty()) { return; }

    const int64_t B = std::max<int64_t>(1, sObj.results.mcRuns);
    std::vector<PhaseTraceInfo>                     phaseTraces;
    std::vector<std::vector<std::vector<SfqEvent>>> trialEvents;
    extract_trial_phase_events(iObj, mObj, sObj, phaseTraces, trialEvents);

    auto sanitize_id = [](const std::string& raw) -> std::string {
        std::string out;
        out.reserve(raw.size());
        for (char c : raw) {
            if (std::isalnum(static_cast<unsigned char>(c))) {
                out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
            } else if (!out.empty() && out.back() != '_') {
                out.push_back('_');
            }
        }
        while (!out.empty() && out.back() == '_') { out.pop_back(); }
        if (out.rfind("PARAM_", 0) == 0 && out.size() > 6) { out = out.substr(6); }
        return out.empty() ? "TGT" : out;
    };

    // Compute per-trial SFQ event timing latency (mean SFQ event timestamp + signed shift from nominal trial 0)
    std::vector<double> trialSfqLatency(static_cast<size_t>(B), std::numeric_limits<double>::quiet_NaN());
    double              nomMeanEvTime  = 0.0;
    int64_t             nomTotalPulses = 0;
    if (!phaseTraces.empty() && !trialEvents.empty()) {
        const auto& nomEvs = trialEvents[0];
        double      sumT0  = 0.0;
        for (const auto& evs : nomEvs) {
            for (const auto& ev : evs) {
                sumT0 += ev.time;
                ++nomTotalPulses;
            }
        }
        if (nomTotalPulses > 0) {
            nomMeanEvTime      = sumT0 / static_cast<double>(nomTotalPulses);
            trialSfqLatency[0] = nomMeanEvTime;
            for (int64_t b = 1; b < B && static_cast<size_t>(b) < trialEvents.size(); ++b) {
                const auto& curEvs = trialEvents[static_cast<size_t>(b)];
                double      sumDt  = 0.0;
                int64_t     nMatch = 0;
                for (size_t k = 0; k < phaseTraces.size(); ++k) {
                    const size_t nPair = std::min(nomEvs[k].size(), curEvs[k].size());
                    for (size_t i = 0; i < nPair; ++i) {
                        if (nomEvs[k][i].polarity == curEvs[k][i].polarity) {
                            sumDt += (curEvs[k][i].time - nomEvs[k][i].time);
                            ++nMatch;
                        }
                    }
                }
                if (nMatch > 0) {
                    trialSfqLatency[static_cast<size_t>(b)] = nomMeanEvTime + (sumDt / static_cast<double>(nMatch));
                }
            }
        }
    }

    // Resolve primary metric for sensitivity & centering
    std::string         metricName = "SFQ_LATENCY";
    std::string         metricUnit = "s";
    std::vector<double> metricTrials = trialSfqLatency;
    bool                hasGoal    = false;
    double              goalVal    = 0.0;
    bool                hasMinGoal = false;
    double              minGoalVal = 0.0;
    bool                hasMaxGoal = false;
    double              maxGoalVal = 0.0;

    std::string reqMetric;
    for (const auto& ss : iObj.sensSpecs) {
        if (!ss.metric.empty()) { reqMetric = ss.metric; }
        if (ss.hasGoal) {
            hasGoal = true;
            goalVal = ss.goal;
        }
    }

    auto apply_meas_as_metric = [&](size_t mIdx) {
        const auto& ms = iObj.measurements[mIdx];
        metricName     = ms.name;
        if (ms.kind == MeasKind::TRIG_TARG || ms.kind == MeasKind::WHEN || ms.kind == MeasKind::MIN_AT
            || ms.kind == MeasKind::MAX_AT || ms.kind == MeasKind::FWHM || ms.kind == MeasKind::DT_PI) {
            metricUnit = "s";
        } else if (ms.kind == MeasKind::COUNT) {
            metricUnit = "pulses";
        } else if (ms.kind == MeasKind::FREQ) {
            metricUnit = "Hz";
        } else if (ms.kind == MeasKind::VPEAK) {
            metricUnit = "V";
        } else if (ms.kind == MeasKind::AREA) {
            metricUnit = "V*s";
        } else {
            metricUnit = "-";
        }
        metricTrials = measTrialValues[mIdx];
        if (ms.hasMinGoal) {
            hasMinGoal = true;
            minGoalVal = ms.minGoal;
        }
        if (ms.hasMaxGoal) {
            hasMaxGoal = true;
            maxGoalVal = ms.maxGoal;
        }
        if (!hasGoal) {
            if (ms.hasValGoal) {
                hasGoal = true;
                goalVal = ms.valGoal;
            } else if (ms.hasMinGoal && ms.hasMaxGoal) {
                hasGoal = true;
                goalVal = 0.5 * (ms.minGoal + ms.maxGoal);
            }
        }
    };

    bool resolvedMetric = false;
    if (!reqMetric.empty()) {
        std::string upReq = reqMetric;
        std::transform(upReq.begin(), upReq.end(), upReq.begin(), ::toupper);
        for (size_t mIdx = 0; mIdx < iObj.measurements.size() && mIdx < measTrialValues.size(); ++mIdx) {
            std::string mUp = iObj.measurements[mIdx].name;
            std::transform(mUp.begin(), mUp.end(), mUp.begin(), ::toupper);
            if (mUp == upReq) {
                apply_meas_as_metric(mIdx);
                resolvedMetric = true;
                break;
            }
        }
        if (!resolvedMetric && upReq != "LATENCY" && upReq != "DELAY" && upReq != "SFQ_LATENCY" && upReq != "SFQ") {
            std::string inner = upReq;
            if (inner.size() > 3 && (inner.rfind("P(", 0) == 0 || inner.rfind("V(", 0) == 0) && inner.back() == ')') {
                inner = inner.substr(2, inner.size() - 3);
            }
            std::replace(inner.begin(), inner.end(), '.', '|');
            for (size_t k = 0; k < phaseTraces.size(); ++k) {
                std::string pLbl = normalize_event_trace_label(phaseTraces[k].label);
                if (pLbl == inner || (pLbl.find('|') != std::string::npos && pLbl.substr(0, pLbl.find('|')) == inner)) {
                    metricName = "LAT(" + pLbl + ")";
                    metricUnit = "s";
                    metricTrials.assign(static_cast<size_t>(B), std::numeric_limits<double>::quiet_NaN());
                    for (int64_t b = 0; b < B && static_cast<size_t>(b) < trialEvents.size(); ++b) {
                        const auto& evs = trialEvents[static_cast<size_t>(b)][k];
                        if (!evs.empty()) {
                            double s = 0.0;
                            for (const auto& e : evs) { s += e.time; }
                            metricTrials[static_cast<size_t>(b)] = s / static_cast<double>(evs.size());
                        }
                    }
                    resolvedMetric = true;
                    break;
                }
            }
        }
    }
    if (!resolvedMetric) {
        for (size_t mIdx = 0; mIdx < iObj.measurements.size() && mIdx < measTrialValues.size(); ++mIdx) {
            if (iObj.measurements[mIdx].kind != MeasKind::PARAM && !measTrialValues[mIdx].empty()
                && std::isfinite(measTrialValues[mIdx][0])) {
                apply_meas_as_metric(mIdx);
                resolvedMetric = true;
                break;
            }
        }
    }

    struct SensRow {
        size_t      targetIdx         = 0;
        std::string name;
        std::string category;
        double      nominalValue      = 1.0;
        double      delta             = 0.05;
        double      valMinus          = 0.0;
        double      valNom            = 0.0;
        double      valPlus           = 0.0;
        double      relSens           = 0.0; // dM / ds
        double      pctSens           = 0.0; // dM per +1% scale
        double      physSens          = 0.0; // dM / dp
        double      normSens          = 0.0; // (p/M) dM/dp
        double      curvature         = 0.0; // d^2M / ds^2
        double      latSens           = 0.0; // dt_SFQ / ds (s per 100%)
        double      weightPct         = 0.0; // variance contribution %
        double      minPassScale      = 1.0;
        double      maxPassScale      = 1.0;
        double      critMarginPct     = 0.0;
        double      centeredMarginPct = 0.0;
        double      centerScale       = 1.0;
        double      centerVal         = 1.0;
        double      centerShiftPct    = 0.0;
        std::string centerMode        = "NOMINAL";
        bool        hasPassBand       = false;
    };

    auto eval_deriv = [](const std::vector<double>& yTrials,
                         size_t                     bMinus,
                         size_t                     bNom,
                         size_t                     bPlus,
                         double                     sMinus,
                         double                     sPlus,
                         double&                    outCurv) -> double {
        outCurv        = 0.0;
        const double y0 = (bNom < yTrials.size()) ? yTrials[bNom] : std::numeric_limits<double>::quiet_NaN();
        const double ym = (bMinus < yTrials.size()) ? yTrials[bMinus] : std::numeric_limits<double>::quiet_NaN();
        const double yp = (bPlus < yTrials.size()) ? yTrials[bPlus] : std::numeric_limits<double>::quiet_NaN();
        if (std::isfinite(ym) && std::isfinite(yp) && sPlus > sMinus + 1e-12) {
            const double g = (yp - ym) / (sPlus - sMinus);
            if (std::isfinite(y0) && sPlus > 1.0 + 1e-9 && sMinus < 1.0 - 1e-9) {
                const double gp = (yp - y0) / (sPlus - 1.0);
                const double gm = (y0 - ym) / (1.0 - sMinus);
                outCurv         = (gp - gm) / (0.5 * (sPlus - sMinus));
            }
            return g;
        }
        if (std::isfinite(yp) && std::isfinite(y0) && std::abs(sPlus - 1.0) > 1e-12) {
            return (yp - y0) / (sPlus - 1.0);
        }
        if (std::isfinite(ym) && std::isfinite(y0) && std::abs(1.0 - sMinus) > 1e-12) {
            return (y0 - ym) / (1.0 - sMinus);
        }
        return 0.0;
    };

    bool anyExplicitCenter = iObj.argMargins;
    for (const auto& ss : iObj.sensSpecs) {
        if (ss.center) { anyExplicitCenter = true; }
    }

    std::vector<SensRow> rows;
    rows.reserve(iObj.marginTargets.size());
    double sumSqSens = 0.0;

    for (size_t tIdx = 0; tIdx < iObj.marginTargets.size(); ++tIdx) {
        const auto&  mt = iObj.marginTargets[tIdx];
        const size_t P  = mt.values.size();
        if (P == 0 || mt.trialIndices.size() != P) { continue; }

        const size_t idxMinus = std::min(mt.minusPointIdx, P - 1);
        const size_t idxNom   = std::min(mt.nomPointIdx, P - 1);
        const size_t idxPlus  = std::min(mt.plusPointIdx, P - 1);

        const size_t bMinus = static_cast<size_t>(std::max<int64_t>(0, mt.trialIndices[idxMinus]));
        const size_t bNom   = static_cast<size_t>(std::max<int64_t>(0, mt.trialIndices[idxNom]));
        const size_t bPlus  = static_cast<size_t>(std::max<int64_t>(0, mt.trialIndices[idxPlus]));

        const double sMinus = mt.values[idxMinus];
        const double sPlus  = mt.values[idxPlus];

        SensRow r;
        r.targetIdx    = tIdx;
        r.name         = mt.displayName;
        r.category     = mt.category;
        r.nominalValue = mt.nominalValue;
        r.delta        = mt.delta;
        r.valMinus     = (bMinus < metricTrials.size()) ? metricTrials[bMinus] : std::numeric_limits<double>::quiet_NaN();
        r.valNom       = (bNom < metricTrials.size()) ? metricTrials[bNom] : std::numeric_limits<double>::quiet_NaN();
        r.valPlus      = (bPlus < metricTrials.size()) ? metricTrials[bPlus] : std::numeric_limits<double>::quiet_NaN();

        r.relSens  = eval_deriv(metricTrials, bMinus, bNom, bPlus, sMinus, sPlus, r.curvature);
        r.pctSens  = 0.01 * r.relSens;
        r.physSens = (std::abs(r.nominalValue) > 1e-30) ? (r.relSens / r.nominalValue) : r.relSens;
        r.normSens = (std::isfinite(r.valNom) && std::abs(r.valNom) > 1e-30) ? (r.relSens / r.valNom) : 0.0;

        double dummyCurv = 0.0;
        r.latSens        = eval_deriv(trialSfqLatency, bMinus, bNom, bPlus, sMinus, sPlus, dummyCurv);

        // Pass/fail evaluation across mt.values
        std::vector<bool> ptPassed(P, true);
        for (size_t p = 0; p < P; ++p) {
            const int64_t b = mt.trialIndices[p];
            if (b == 0) {
                ptPassed[p] = true;
                continue;
            }
            bool ok = true;
            if (!phaseTraces.empty() && static_cast<size_t>(b) < trialEvents.size()) {
                const auto diag = compare_event_sequences(
                        phaseTraces, trialEvents[0], trialEvents[static_cast<size_t>(b)], iObj.argMarginTol, iObj.transSim.tstop());
                if (!diag.passed) { ok = false; }
            }
            if (ok && (hasMinGoal || hasMaxGoal) && static_cast<size_t>(b) < metricTrials.size()) {
                const double mv = metricTrials[static_cast<size_t>(b)];
                if (!std::isfinite(mv)) {
                    ok = false;
                } else {
                    if (hasMinGoal && mv < minGoalVal) { ok = false; }
                    if (hasMaxGoal && mv > maxGoalVal) { ok = false; }
                }
            }
            ptPassed[p] = ok;
        }

        size_t loIdx = idxNom;
        size_t hiIdx = idxNom;
        while (loIdx > 0 && ptPassed[loIdx - 1]) { --loIdx; }
        while (hiIdx + 1 < P && ptPassed[hiIdx + 1]) { ++hiIdx; }

        r.minPassScale = mt.values[loIdx];
        r.maxPassScale = mt.values[hiIdx];

        if (P > 3) {
            r.hasPassBand       = true;
            const double mLow   = (1.0 - r.minPassScale) * 100.0;
            const double mHigh  = (r.maxPassScale - 1.0) * 100.0;
            r.critMarginPct     = std::min(mLow, mHigh);
            r.centeredMarginPct = 0.5 * (r.maxPassScale - r.minPassScale) * 100.0;
            r.centerScale       = 0.5 * (r.minPassScale + r.maxPassScale);
            r.centerMode        = "PASS_BAND";
            if (hasGoal && !anyExplicitCenter && std::isfinite(r.valNom) && std::abs(r.relSens) > 1e-30) {
                r.centerScale = std::clamp(1.0 - (r.valNom - goalVal) / r.relSens, r.minPassScale, r.maxPassScale);
                r.centerMode  = "GOAL_NEWTON";
            }
        } else {
            if (hasGoal && std::isfinite(r.valNom) && std::abs(r.relSens) > 1e-30) {
                r.centerScale = std::clamp(1.0 - (r.valNom - goalVal) / r.relSens, 0.2, 3.0);
                r.centerMode  = "GOAL_NEWTON";
                if (hasMinGoal && hasMaxGoal) {
                    const double sa = std::clamp(1.0 + (minGoalVal - r.valNom) / r.relSens, 0.0, 3.0);
                    const double sb = std::clamp(1.0 + (maxGoalVal - r.valNom) / r.relSens, 0.0, 3.0);
                    r.minPassScale  = std::min(sa, sb);
                    r.maxPassScale  = std::max(sa, sb);
                    r.critMarginPct = std::min((1.0 - r.minPassScale) * 100.0, (r.maxPassScale - 1.0) * 100.0);
                    r.centeredMarginPct = 0.5 * (r.maxPassScale - r.minPassScale) * 100.0;
                    r.hasPassBand       = true;
                }
            } else if (loIdx == idxNom && hiIdx > idxNom) {
                r.centerScale = 1.0 + 0.5 * r.delta;
                r.centerMode  = "BOUND_SHIFT";
            } else if (hiIdx == idxNom && loIdx < idxNom) {
                r.centerScale = 1.0 - 0.5 * r.delta;
                r.centerMode  = "BOUND_SHIFT";
            } else {
                r.centerScale = 1.0;
                r.centerMode  = "NOMINAL";
            }
            if (!r.hasPassBand) {
                const double effLatSens = (std::abs(r.latSens) > 1e-18) ? std::abs(r.latSens)
                                                                        : ((metricUnit == "s") ? std::abs(r.relSens) : 0.0);
                if (effLatSens > 1e-18) {
                    r.critMarginPct     = std::min(100.0, (iObj.argMarginTol / effLatSens) * 100.0);
                    r.centeredMarginPct = r.critMarginPct;
                } else {
                    r.critMarginPct     = r.delta * 100.0;
                    r.centeredMarginPct = r.critMarginPct;
                }
            }
        }

        r.centerVal      = r.centerScale * r.nominalValue;
        r.centerShiftPct = (r.centerScale - 1.0) * 100.0;
        sumSqSens += r.relSens * r.relSens;
        rows.push_back(std::move(r));
    }

    for (auto& r : rows) {
        r.weightPct = (sumSqSens > 1e-60) ? ((r.relSens * r.relSens) / sumSqSens * 100.0) : 0.0;
    }

    std::stable_sort(rows.begin(), rows.end(), [](const SensRow& a, const SensRow& b) {
        if (std::abs(std::abs(a.relSens) - std::abs(b.relSens)) > 1e-30) {
            return std::abs(a.relSens) > std::abs(b.relSens);
        }
        return a.targetIdx < b.targetIdx;
    });

    // Export SENS_* and CENTER_* parameters into iObj.parameters
    auto& paramMap  = const_cast<Input&>(iObj).parameters;
    auto  set_param = [&](const std::string& name, double val) {
        if (!std::isfinite(val)) { return; }
        Parameter p;
        p.set_expression(Misc::precise_to_string(val));
        p.set_value(val);
        paramMap[ParameterName(name, std::nullopt)] = p;
    };

    double maxAbsSens  = 0.0;
    double maxNormSens = 0.0;
    for (const auto& r : rows) {
        const std::string id = sanitize_id(r.name);
        set_param("SENS_" + id, r.relSens);
        set_param("SENS_PCT_" + id, r.pctSens);
        set_param("SENS_NORM_" + id, r.normSens);
        set_param("SENS_CURV_" + id, r.curvature);
        set_param("SENS_WEIGHT_" + id, r.weightPct);
        set_param("SENS_LAT_" + id, r.latSens);
        set_param("CENTER_SCALE_" + id, r.centerScale);
        set_param("CENTER_VAL_" + id, r.centerVal);
        set_param("CENTER_SHIFT_PCT_" + id, r.centerShiftPct);
        set_param("MARGIN_CRIT_PCT_" + id, r.critMarginPct);
        set_param("MARGIN_CENTERED_PCT_" + id, r.centeredMarginPct);
        maxAbsSens  = std::max(maxAbsSens, std::abs(r.relSens));
        maxNormSens = std::max(maxNormSens, std::abs(r.normSens));
    }
    set_param("SENS_MAX_ABS", maxAbsSens);
    set_param("SENS_MAX_NORM", maxNormSens);
    set_param("SENS_RSS", std::sqrt(sumSqSens));

    const bool isTimeMetric = (metricUnit == "s");
    const double dispScale  = isTimeMetric ? 1e12 : 1.0;
    const std::string dispUnit = isTimeMetric ? "ps" : metricUnit;

    if (!iObj.argMin && !rows.empty()) {
        const double nomValDisp = std::isfinite(rows.front().valNom) ? (rows.front().valNom * dispScale) : 0.0;
        std::cout << "\n===================================================================================================================\n";
        std::cout << "JoSIM Batched OAAT Sensitivity & Margin Centering Report (" << rows.size() << " target(s), "
                  << B << " batched trials, metric = " << metricName << " [" << std::fixed << std::setprecision(3)
                  << nomValDisp << " " << dispUnit << "])\n";
        std::cout << "===================================================================================================================\n";
        std::cout << std::left << std::setw(5) << "Rank" << std::setw(16) << "Target" << std::setw(7) << "Cat"
                  << std::right << std::setw(12) << "Nominal" << std::setw(14) << ("dM/ds(" + dispUnit + ")")
                  << std::setw(12) << "Norm(S_k)" << std::setw(9) << "Var(%)" << std::setw(12) << "Center[x]"
                  << std::setw(12) << "CenterVal" << std::setw(10) << "Shift(%)" << "  Margin(Crit->Cent)\n";
        std::cout << "-------------------------------------------------------------------------------------------------------------------\n";

        for (size_t idx = 0; idx < rows.size(); ++idx) {
            const auto&       r = rows[idx];
            std::ostringstream margStr;
            margStr << std::fixed << std::setprecision(1) << r.critMarginPct << "% -> " << r.centeredMarginPct << "% ("
                    << r.centerMode << ")";
            std::ostringstream shiftStr;
            shiftStr << (r.centerShiftPct >= 0.0 ? "+" : "") << std::fixed << std::setprecision(1) << r.centerShiftPct
                     << "%";

            std::cout << std::left << std::setw(5) << (idx + 1) << std::setw(16) << r.name.substr(0, 15) << std::setw(7)
                      << r.category.substr(0, 6) << std::right << std::scientific << std::setprecision(3)
                      << std::setw(12) << r.nominalValue << std::fixed << std::setprecision(3) << std::setw(14)
                      << (r.relSens * dispScale) << std::setw(12) << r.normSens << std::setprecision(1) << std::setw(9)
                      << r.weightPct << std::setprecision(3) << std::setw(12) << r.centerScale << std::scientific
                      << std::setprecision(3) << std::setw(12) << r.centerVal << std::setw(10) << shiftStr.str() << "  "
                      << std::left << margStr.str() << "\n";
        }
        std::cout << "===================================================================================================================\n";
    }

    if (iObj.sensFile.has_value() && !iObj.sensFile->empty()) {
        std::ofstream ofs(*iObj.sensFile);
        if (ofs.is_open()) {
            const std::string& fn     = *iObj.sensFile;
            const bool         isJson = (fn.size() >= 5
                                 && (fn.substr(fn.size() - 5) == ".json" || fn.substr(fn.size() - 5) == ".JSON"));
            if (isJson) {
                ofs << "{\n  \"metric\": \"" << metricName << "\",\n"
                    << "  \"unit\": \"" << metricUnit << "\",\n"
                    << "  \"trials\": " << B << ",\n"
                    << "  \"targets\": [\n";
                ofs << std::scientific << std::setprecision(12);
                for (size_t i = 0; i < rows.size(); ++i) {
                    const auto& r = rows[i];
                    ofs << "    {\n"
                        << "      \"rank\": " << (i + 1) << ",\n"
                        << "      \"target\": \"" << r.name << "\",\n"
                        << "      \"category\": \"" << r.category << "\",\n"
                        << "      \"nominal_value\": " << r.nominalValue << ",\n"
                        << "      \"delta\": " << r.delta << ",\n"
                        << "      \"rel_sens\": " << r.relSens << ",\n"
                        << "      \"pct_sens\": " << r.pctSens << ",\n"
                        << "      \"phys_sens\": " << r.physSens << ",\n"
                        << "      \"norm_sens\": " << r.normSens << ",\n"
                        << "      \"curvature\": " << r.curvature << ",\n"
                        << "      \"lat_sens_s\": " << r.latSens << ",\n"
                        << "      \"variance_weight_pct\": " << r.weightPct << ",\n"
                        << "      \"min_pass_scale\": " << r.minPassScale << ",\n"
                        << "      \"max_pass_scale\": " << r.maxPassScale << ",\n"
                        << "      \"crit_margin_pct\": " << r.critMarginPct << ",\n"
                        << "      \"centered_margin_pct\": " << r.centeredMarginPct << ",\n"
                        << "      \"center_scale\": " << r.centerScale << ",\n"
                        << "      \"center_value\": " << r.centerVal << ",\n"
                        << "      \"center_shift_pct\": " << r.centerShiftPct << ",\n"
                        << "      \"center_mode\": \"" << r.centerMode << "\"\n"
                        << "    }" << (i + 1 < rows.size() ? "," : "") << "\n";
                }
                ofs << "  ]\n}\n";
            } else {
                ofs << "rank,target,category,nominal_value,delta,rel_sens,pct_sens,phys_sens,norm_sens,curvature,"
                       "lat_sens_s,variance_weight_pct,min_pass_scale,max_pass_scale,crit_margin_pct,"
                       "centered_margin_pct,center_scale,center_value,center_shift_pct,center_mode\n";
                ofs << std::scientific << std::setprecision(12);
                for (size_t i = 0; i < rows.size(); ++i) {
                    const auto& r = rows[i];
                    ofs << (i + 1) << ",\"" << r.name << "\",\"" << r.category << "\"," << r.nominalValue << ","
                        << r.delta << "," << r.relSens << "," << r.pctSens << "," << r.physSens << "," << r.normSens
                        << "," << r.curvature << "," << r.latSens << "," << r.weightPct << "," << r.minPassScale << ","
                        << r.maxPassScale << "," << r.critMarginPct << "," << r.centeredMarginPct << ","
                        << r.centerScale << "," << r.centerVal << "," << r.centerShiftPct << "," << r.centerMode
                        << "\n";
                }
            }
        }
    }
}

