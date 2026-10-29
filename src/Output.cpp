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
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
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
        if (iObj.argEventsOnly) { return; }
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
    if (canStreamDirect && !mObj.relevantTraces.empty()) {
        const size_t tSize = sObj.results.timeAxis.size();
        for (const auto& tr : mObj.relevantTraces) {
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
        cols.reserve(1 + (!mObj.relevantTraces.empty() ? mObj.relevantTraces.size() : mObj.nm.size()));
        cols.push_back({"time", 'T', -1, 0, nullptr, nullptr, -1});
        traces.emplace_back("time");
        traces.back().type_ = 'T';

        if (!mObj.relevantTraces.empty()) {
            for (const auto& i : mObj.relevantTraces) {
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
    if (!iObj.cli_output_file && iObj.output_files.empty()) { format_cout(iObj.argMin); }
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
    // Print only the indices of the relevant traces
    int64_t cc = 0;
    if (mObj.relevantTraces.size() != 0 && sObj.results.mcRuns > 1 && !sObj.results.mcTraceOut.empty()) {
        const int64_t B          = sObj.results.mcRuns;
        const int64_t nRows      = sObj.results.mcNumRows;
        const int64_t nKept      = sObj.results.mcNumKeptSteps;
        const int64_t nTraces    = static_cast<int64_t>(mObj.relevantTraces.size());
        const size_t  nPrint     = result_indices.size();
        const int64_t colsPerTr  = 4 + (iObj.argMcAll ? B : 0);

        auto make_mc_label = [](const std::string& label, const std::string& suffix) -> std::string {
            if (suffix.empty()) { return label; }
            if (label.size() >= 2 && label.front() == '"' && label.back() == '"') {
                return label.substr(0, label.size() - 1) + suffix + "\"";
            }
            return label + suffix;
        };

        for (int64_t idx = 0; idx < nTraces; ++idx) {
            const auto&       i     = mObj.relevantTraces[idx];
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
                traces.back().data_.resize(nPrint, 0.0);
            }
            if (iObj.argMcAll) {
                for (int64_t b = 0; b < B; ++b) {
                    traces.emplace_back(make_mc_label(label, "_MC" + std::to_string(b)));
                    traces.back().fileIndex = i.fIndex;
                    traces.back().type_     = tChar;
                    traces.back().data_.resize(nPrint, 0.0);
                }
            }
        }

        struct McStat {
            double peakMean = 0.0;
            double peakStd  = 0.0;
            double peakMin  = 0.0;
            double peakMax  = 0.0;
        };
        std::vector<McStat> stats(nTraces);

        auto process_mc_trace = [&](int64_t idx) {
            const auto&   i        = mObj.relevantTraces[idx];
            const auto&   st       = i.storageType;
            const int64_t baseSlot = 1 + idx * colsPerTr;
            auto&         trMean   = traces[baseSlot + 0].data_;
            auto&         trStd    = traces[baseSlot + 1].data_;
            auto&         trMin    = traces[baseSlot + 2].data_;
            auto&         trMax    = traces[baseSlot + 3].data_;

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

            std::vector<double> meanVec(nPrint, 0.0);
            std::vector<double> m2Vec(nPrint, 0.0);
            std::vector<double> vMin(nPrint, 1e300);
            std::vector<double> vMax(nPrint, -1e300);
            std::vector<double> trialWave(nPrint, 0.0);
            std::vector<double> vectphase;
            if (st == StorageType::Phase && iObj.argAnal == AnalysisType::Voltage
                && !(i.deviceLabel.value().at(3) == 'B' && sv >= 0)) {
                vectphase.resize(xsize, 0.0);
            }

            Function localSrc;
            bool     isISrc = (st == StorageType::Current && i.deviceLabel.value().at(3) == 'I');
            if (isISrc) { localSrc = mObj.sourcegen.at(i.sourceIndex.value()); }

            for (int64_t b = 0; b < B; ++b) {
                const double* p1 = (s1 >= 0) ? (sObj.results.mcTraceOut.data() + (b * nRows + s1) * nKept) : nullptr;
                const double* p2 = (s2 >= 0) ? (sObj.results.mcTraceOut.data() + (b * nRows + s2) * nKept) : nullptr;
                const double* pv = (sv >= 0) ? (sObj.results.mcTraceOut.data() + (b * nRows + sv) * nKept) : nullptr;

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

                if (iObj.argMcAll) { traces[baseSlot + 4 + b].data_ = trialWave; }
                const double invCount = 1.0 / static_cast<double>(b + 1);
                for (size_t pIdx = 0; pIdx < nPrint; ++pIdx) {
                    const double v      = trialWave[pIdx];
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

        if (!iObj.argMin && nTraces > 0) {
            std::cout << "\nJoSIM Monte Carlo Robustness Summary (" << B << " trials):\n";
            std::cout << "  " << std::left << std::setw(22) << "Trace" << std::right << std::setw(14) << "Peak Mean"
                      << std::setw(14) << "Peak 1-Sigma" << std::setw(14) << "Peak Min" << std::setw(14) << "Peak Max"
                      << std::setw(12) << "CV (%)" << "\n";
            const int64_t maxShow = std::min<int64_t>(nTraces, 16);
            for (int64_t idx = 0; idx < maxShow; ++idx) {
                const auto&  st = stats[idx];
                const double cv = (std::abs(st.peakMean) > 1e-18)
                                          ? (100.0 * st.peakStd / std::abs(st.peakMean))
                                          : 0.0;
                std::cout << "  " << std::left << std::setw(22) << mObj.relevantTraces[idx].deviceLabel.value()
                          << std::right << std::scientific << std::setprecision(4) << std::setw(14) << st.peakMean
                          << std::setw(14) << st.peakStd << std::setw(14) << st.peakMin << std::setw(14) << st.peakMax
                          << std::fixed << std::setprecision(2) << std::setw(11) << cv << "%\n";
            }
            if (nTraces > maxShow) {
                std::cout << "  ... (" << (nTraces - maxShow) << " additional traces written to output)\n";
            }
            std::cout << "\n";
        }
    } else if (mObj.relevantTraces.size() != 0) {
        // Create progress bar
        ProgressBar bar;
        if (!iObj.argMin) {
            bar.create_thread();
            bar.set_bar_width(30);
            bar.fill_bar_progress_with("O");
            bar.fill_bar_remainder_with(" ");
            bar.set_status_text("Formatting Output");
            bar.set_total((float) mObj.relevantTraces.size());
        }
        const int64_t nTraces = static_cast<int64_t>(mObj.relevantTraces.size());
        for (int64_t idx = 0; idx < nTraces; ++idx) {
            const auto& i = mObj.relevantTraces[idx];
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
        auto process_trace = [&](int64_t idx) {
            const auto& i        = mObj.relevantTraces[idx];
            auto&       outTrace = traces[idx + 1];
            auto&       st       = i.storageType;
            if (!iObj.argMin && idx % 4 == 0) { bar.update(static_cast<float>(idx)); }
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
        if (iObj.argPar && nTraces >= 2) {
            const int nThreads = std::min<int>(omp_get_max_threads(), static_cast<int>(nTraces));
#pragma omp parallel for schedule(dynamic) num_threads(nThreads)
            for (int64_t idx = 0; idx < nTraces; ++idx) { process_trace(idx); }
        } else
#endif
        {
            for (int64_t idx = 0; idx < nTraces; ++idx) { process_trace(idx); }
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
        for (size_t i = 0; i < numCols; ++i) { colPtrs[i] = traces[tIndices[i]].data_.data(); }
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
        outfile.write(reinterpret_cast<const char*>(traces[tIndices[c]].data_.data()), colBytes);
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
    for (size_t c = 0; c < numCols; ++c) { colPtrs[c] = traces[tIndices[c]].data_.data(); }

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
    for (size_t c = 0; c < numCols; ++c) { colPtrs[c] = traces[tIndices[c]].data_.data(); }

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
                    outfile << std::left << std::setw(pointSizeSpacing) << "" << traces.at(tIndices.at(j)).data_.at(i)
                            << "\n";
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
                std::cout << std::setw(15) << std::scientific << std::setprecision(6) << traces.at(i).data_.at(j)
                          << " ";
            }
            std::cout << std::setw(15) << std::scientific << std::setprecision(6)
                      << traces.at(traces.size() - 1).data_.at(j) << "\n";
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
